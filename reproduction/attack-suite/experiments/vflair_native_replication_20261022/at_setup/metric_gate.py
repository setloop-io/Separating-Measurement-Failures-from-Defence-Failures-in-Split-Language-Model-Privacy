"""Nineteen tiny CPU software checks; no model load, generation or attack."""

import argparse
import ast
import collections
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from types import SimpleNamespace


REFERENCE = "def add_values(left, right):\n    total = left + right\n    return total\n"
CHANGED = "def add_values(left, right):\n    total = left - right\n    return 0\n"
SOURCE_SHA256 = "45618d63e76407e9d35d6f7f220b9729903579a9e7966105860bf0661a0bce10"


def evaluate(native_source):
    from codebleu import calc_codebleu
    import codebleu
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    pins = {"codebleu": "0.7.0", "tree-sitter": "0.22.3", "tree-sitter-python": "0.21.0",
            "transformers": "4.39.0", "tokenizers": "0.15.2", "setuptools": "80.9.0", "wheel": "0.45.1"}
    actual = {name: importlib.metadata.version(name) for name in pins}
    if actual != pins:
        raise RuntimeError(f"AT metric stack differs from exact admitted versions: {actual}")
    kwargs = dict(lang="python", weights=(0.25, 0.25, 0.25, 0.25))
    identical = calc_codebleu([REFERENCE], [REFERENCE], **kwargs)
    changed = calc_codebleu([REFERENCE], [CHANGED], **kwargs)

    def mapping_fixture(text):
        return collections.UserDict({"input_ids": [ord(x) for x in text], "attention_mask": [1] * len(text)})

    mapped_same = calc_codebleu([REFERENCE], [REFERENCE], tokenizer=mapping_fixture, **kwargs)
    mapped_changed = calc_codebleu([REFERENCE], [CHANGED], tokenizer=mapping_fixture, **kwargs)
    ten = [calc_codebleu([REFERENCE], [REFERENCE], **kwargs) for _ in range(10)]
    tests = {
        "default_identical_code_scores_one": abs(identical["codebleu"] - 1) < 1e-12,
        "default_identical_all_components_one": all(abs(value - 1) < 1e-12 for value in identical.values()),
        "changed_code_scores_lower": changed["codebleu"] < identical["codebleu"],
        "default_components_finite": all(math.isfinite(value) for row in (identical, changed) for value in row.values()),
        "ten_repeated_python_calls_identical": all(row == ten[0] for row in ten),
        "ten_repeated_python_max_equals_single": max(row["codebleu"] for row in ten) == identical["codebleu"],
        "mapping_fixture_ngram_ignores_code_change": mapped_same["ngram_match_score"] == mapped_changed["ngram_match_score"],
        "mapping_fixture_weighted_ngram_ignores_code_change": mapped_same["weighted_ngram_match_score"] == mapped_changed["weighted_ngram_match_score"],
    }
    vocabulary = {"[UNK]": 0, "def": 1, "add_values": 2, "left": 3, "right": 4, "total": 5,
                  "return": 6, "+": 7, "-": 8, "0": 9, "=": 10, "(": 11, "):": 12,
                  ",": 13, ")": 14, ":": 15}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                model_input_names=["input_ids", "attention_mask"])
    encoded_ref, encoded_changed = hf(REFERENCE), hf(CHANGED)
    native_same = calc_codebleu([REFERENCE], [REFERENCE], tokenizer=hf, **kwargs)
    native_changed = calc_codebleu([REFERENCE], [CHANGED], tokenizer=hf, **kwargs)
    token_same = calc_codebleu([REFERENCE], [REFERENCE], tokenizer=hf.tokenize, **kwargs)
    token_changed = calc_codebleu([REFERENCE], [CHANGED], tokenizer=hf.tokenize, **kwargs)
    tests.update({
        "actual_hf_returns_batchencoding": type(encoded_ref).__name__ == "BatchEncoding",
        "actual_hf_iteration_yields_native_llama_field_names": list(encoded_ref) == ["input_ids", "attention_mask"],
        "code_change_changes_actual_token_ids": encoded_ref["input_ids"] != encoded_changed["input_ids"],
        "native_hf_identical_code_not_one": native_same["codebleu"] < 1,
        "native_hf_ngram_ignores_code_change": native_same["ngram_match_score"] == native_changed["ngram_match_score"],
        "native_hf_weighted_ngram_ignores_code_change": native_same["weighted_ngram_match_score"] == native_changed["weighted_ngram_match_score"],
        "token_sequence_callback_identical_one_diagnostic_only": abs(token_same["codebleu"] - 1) < 1e-12,
        "token_sequence_callback_changes_lexical_score_diagnostic_only": token_same["ngram_match_score"] != token_changed["ngram_match_score"],
    })
    raw = Path(native_source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Native utility source differs from the pinned revision")
    nodes = [node for node in ast.walk(ast.parse(raw.decode())) if isinstance(node, ast.FunctionDef) and node.name == "generate_assessment"]
    if len(nodes) != 1:
        raise RuntimeError("Native utility function is not unique")
    namespace = {}
    exec(compile(ast.Module(body=[copy.deepcopy(nodes[0])], type_ignores=[]), str(native_source), "exec"), namespace)

    class DecodeFixture:
        def __call__(self, text):
            return hf(text)

        def decode(self, ids, **options):
            return [REFERENCE, CHANGED][ids[0]]

    top = SimpleNamespace(args=SimpleNamespace(model_architect="CLM", task_type="CausalLM", dataset="CodeAlpaca",
                                               max_new_tokens=200, tokenizer=DecodeFixture()))
    original_calc = codebleu.calc_codebleu
    calls = []

    def record_call(*args, **options):
        result = original_calc(*args, **options)
        calls.append({"lang": options["lang"], "result": result})
        return result

    codebleu.calc_codebleu = record_call
    try:
        result = namespace["generate_assessment"](top, [[0], [1]], [[0], [0]])
    finally:
        codebleu.calc_codebleu = original_calc
    tests.update({
        "unchanged_native_function_twenty_calls_for_two_rows": len(calls) == 20,
        "unchanged_native_function_all_calls_python": all(row["lang"] == "python" for row in calls),
        "unchanged_native_function_expected_macro_mean": abs(result["acc"] - (native_same["codebleu"] + native_changed["codebleu"]) / 2) < 1e-12,
    })
    file_hashes = []
    for name in ("codebleu", "tree-sitter", "tree-sitter-python"):
        distribution = importlib.metadata.distribution(name)
        for relative in distribution.files or []:
            path = distribution.locate_file(relative)
            if path.is_file() and path.suffix != ".pyc":
                file_hashes.append({"distribution": name, "path": str(path),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    return {"schema": "original-at-metric-software-gate-v1", "status": "PASS_NATIVE_METRIC_QUALIFIED" if all(tests.values()) else "FAILED",
            "checks_passed": sum(tests.values()), "checks_total": len(tests), "checks": tests, "versions": actual,
            "default_identical": identical, "default_changed": changed, "native_hf_identical": native_same,
            "native_hf_changed": native_changed, "native_two_row_result": result, "native_calls": calls,
            "native_source_sha256": SOURCE_SHA256,
            "native_function_ast_sha256": hashlib.sha256(ast.dump(nodes[0], include_attributes=False).encode()).hexdigest(),
            "installed_metric_file_hashes": file_hashes,
            "native_metric_policy": "preserve-upstream-mapping-callback-v1", "native_runtime_modified": False,
            "corrected_metric_executed_on_model_outputs": False, "actual_model_tokenizer_used": False,
            "software_fixture_tokenizer_used": True, "model_constructed_loaded_or_executed": False,
            "qualification": "Native HF mapping callback makes lexical components score field names. Original ten-language loop repeats Python calls. No fallback installed.",
            "software_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.native_source)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "checks_passed", "checks_total", "versions", "qualification")}), flush=True)
    return 0 if result["status"] == "PASS_NATIVE_METRIC_QUALIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
