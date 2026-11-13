"""Llama/CodeAlpaca-only import isolation in a fresh disposable native copy.

No native function/class AST is altered. Removed multimodal/unselected imports
are unavailable; no placeholder models, attacks, parsers or metrics are supplied.
Do not apply after the separate GPT-2 isolation profile.
"""

import ast
import hashlib
import json
from pathlib import Path


PROFILE = "llama-codealpaca-imports-v1"
DROP_MODULES = {
    "src/load/LoadModels.py": {"models.bottom_models", "models.global_models", "models.autoencoder"},
    "src/load/LoadParty.py": {"party.qwen_active_party", "party.qwen_passive_party", "party.paillier_passive_party", "party.paillier_active_party"},
    "src/party/llm_party.py": {"models.llm_models.processors.blip_processors"},
    "src/party/passive_party.py": {"utils.inferdpt_utils", "utils.custext_utils", "utils.santext_utils", "utils.cluster_utils"},
    "src/evaluates/MainTaskVFL_LLM.py": {"utils.marvell_functions", "evaluates.vqa_evaluator", "evaluates.gms8k_evaluator", "evaluates.math_evaluator", "framework.client.DistributedCommunication"},
    "src/party/active_party.py": {"framework.client.DistributedCommunication"},
    "src/models/llm_models/llama.py": {"models.llm_models.minigpt4.minigpt4", "models.llm_models.minigpt4.conversation"},
    "src/load/llm_model_loaders/LlamaModelLoader.py": {"models.llm_models.minigpt4.minigpt4"},
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bodies(tree):
    return [ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]


def prepare(root, manifest_out):
    root, manifest_out = Path(root), Path(manifest_out)
    model_tree = ast.parse((root / "src/load/LoadModels.py").read_text())
    loaders = [node.module for node in model_tree.body if isinstance(node, ast.ImportFrom)
               and (node.module or "").startswith("llm_model_loaders.")]
    if "llm_model_loaders.LlamaModelLoader" not in loaders or len(loaders) < 2:
        raise RuntimeError("Llama isolation requires a fresh native source copy, not the GPT-2-isolated runtime")
    changes = []
    # Retain unrelated legacy/Python-2/incomplete files byte-for-byte. They are
    # not imported by this selected route and need no compatibility parse.
    targets = set(DROP_MODULES) | {
        "src/load/LoadDataset.py", "src/evaluates/attacks/attack_api.py",
        "src/utils/basic_functions.py", "src/utils/communication_protocol_funcs.py",
        "src/dataset/party_dataset.py",
    }
    for path in sorted(root / name for name in targets):
        relative = str(path.relative_to(root))
        original = path.read_text()
        tree = ast.parse(original)
        edits = []
        for node in tree.body:
            remove, replacement = False, None
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                remove = module in DROP_MODULES.get(relative, set())
                if relative == "src/load/LoadModels.py" and module.startswith("llm_model_loaders."):
                    remove = module != "llm_model_loaders.LlamaModelLoader"
                if relative == "src/evaluates/attacks/attack_api.py" and module.startswith("evaluates.attacks."):
                    remove = module.rsplit(".", 1)[-1] not in {"VanillaModelInversion_WhiteBox", "WhiteBoxInversion", "BiSR"}
                if relative == "src/evaluates/MainTaskVFL_LLM.py" and module == "transformers":
                    names = [name for name in node.names if name.name != "Qwen2ForCausalLM"]
                    if len(names) != len(node.names):
                        replacement = ast.unparse(ast.ImportFrom(module=module, names=names, level=node.level)) if names else ""
                if module == "torchvision" or module.startswith("torchvision."):
                    remove = True
            if isinstance(node, ast.Import):
                if any(name.name == "torchvision" or name.name.startswith("torchvision.") for name in node.names):
                    remove = True
            if isinstance(node, ast.Assign):
                names = {target.id for target in node.targets if isinstance(target, ast.Name)}
                if relative == "src/load/LoadModels.py" and "Loader_Map" in names:
                    replacement = "Loader_Map = {'Llama': LlamaModelLoader, 'llama': LlamaModelLoader}"
                if relative == "src/party/llm_party.py" and "PROCESSOR_DICT" in names:
                    replacement = "PROCESSOR_DICT = {}"
                if relative in {"src/load/LoadDataset.py", "src/utils/basic_functions.py"} and names.intersection({"tp", "transform", "transform_fn"}):
                    remove = True
            if remove or replacement is not None:
                edits.append((node.lineno, node.end_lineno, "" if remove else replacement))
        if not edits:
            continue
        before = _sha(path)
        lines = original.splitlines(keepends=True)
        for first, last, replacement in reversed(edits):
            lines[first - 1:last] = [replacement + "\n"] if replacement else ["\n"]
        rewritten = "".join(lines)
        if _bodies(tree) != _bodies(ast.parse(rewritten)):
            raise RuntimeError(f"Import isolation changed a native function/class AST: {relative}")
        path.write_text(rewritten)
        changes.append({"path": relative, "before_sha256": before, "after_sha256": _sha(path),
                        "function_class_ast_preserved": True,
                        "edits": [{"first_line": first, "last_line": last, "replacement": text}
                                  for first, last, text in edits]})
    result = {"profile": PROFILE, "changes": changes,
              "scope": "Top-level imports and unused registries only; every native top-level function/class AST is unchanged.",
              "retained_model_family": "Llama/CodeLlama CLM", "retained_metric": "native CodeBLEU; no metric fallback",
              "runtime_import_or_model_execution_performed_by_prepare": False}
    manifest_out.write_text(json.dumps(result, indent=2) + "\n")
    return changes
