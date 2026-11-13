"""Explicit GPT2-only import isolation in a disposable upstream source copy.

No function or class body is changed. Unselected module registries are narrowed;
removed imports are unavailable, rather than replaced with dummy algorithms.
"""
import ast
import hashlib
import json
from pathlib import Path


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


DROP_MODULES = {
    'src/load/LoadModels.py': ['models.bottom_models', 'models.global_models', 'models.autoencoder'],
    'src/load/LoadParty.py': ['party.qwen_active_party', 'party.qwen_passive_party', 'party.paillier_passive_party', 'party.paillier_active_party'],
    'src/party/llm_party.py': ['models.llm_models.processors.blip_processors'],
    'src/party/passive_party.py': ['utils.inferdpt_utils', 'utils.custext_utils', 'utils.santext_utils', 'utils.cluster_utils'],
    'src/evaluates/MainTaskVFL_LLM.py': ['utils.marvell_functions', 'evaluates.vqa_evaluator', 'evaluates.gms8k_evaluator', 'evaluates.math_evaluator', 'framework.client.DistributedCommunication'],
    'src/party/active_party.py': ['framework.client.DistributedCommunication'],
}


def prepare(root, manifest_out):
    changes = []
    # Parse only the explicit selected-route compatibility targets. The pinned
    # repository also contains unrelated, syntactically incomplete legacy files;
    # they are retained byte-for-byte and are never imported on this route.
    targets = set(DROP_MODULES) | {
        'src/load/LoadDataset.py', 'src/evaluates/attacks/attack_api.py',
        'src/utils/basic_functions.py', 'src/utils/communication_protocol_funcs.py',
        'src/dataset/party_dataset.py'
    }
    for path in sorted(root / name for name in targets):
        rel = str(path.relative_to(root))
        text = path.read_text()
        tree = ast.parse(text)
        spans = []
        for node in tree.body:
            remove = False
            replacement = None
            if isinstance(node, ast.ImportFrom):
                module = node.module or ''
                remove = module in DROP_MODULES.get(rel, [])
                if rel == 'src/load/LoadModels.py' and module.startswith('llm_model_loaders.'):
                    remove = module != 'llm_model_loaders.GPT2ModelLoader'
                if rel == 'src/evaluates/attacks/attack_api.py' and module.startswith('evaluates.attacks.'):
                    remove = module.rsplit('.', 1)[-1] not in ['VanillaModelInversion_WhiteBox', 'WhiteBoxInversion', 'BiSR']
                if rel == 'src/evaluates/MainTaskVFL_LLM.py' and module == 'transformers':
                    names = [n for n in node.names if n.name != 'Qwen2ForCausalLM']
                    if len(names) != len(node.names):
                        replacement = ast.unparse(ast.ImportFrom(module=module, names=names, level=node.level)) if names else ''
                if module == 'torchvision' or module.startswith('torchvision.'):
                    # Only image helpers refer to these names; GPT2 uses none.
                    remove = True
            if isinstance(node, ast.Import):
                if any(n.name == 'torchvision' or n.name.startswith('torchvision.') for n in node.names):
                    remove = True
            if isinstance(node, ast.Assign):
                targetnames = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if rel == 'src/load/LoadModels.py' and 'Loader_Map' in targetnames:
                    replacement = "Loader_Map = {'GPT2': GPT2ModelLoader}"
                if rel == 'src/party/llm_party.py' and 'PROCESSOR_DICT' in targetnames:
                    replacement = 'PROCESSOR_DICT = {}'
                # Top-level image transform construction follows removed torchvision imports.
                if rel in ['src/load/LoadDataset.py', 'src/utils/basic_functions.py'] and any(x in targetnames for x in ['tp', 'transform', 'transform_fn']):
                    remove = True
            if remove or replacement is not None:
                spans.append((node.lineno, node.end_lineno, '' if remove else replacement))
        if not spans:
            continue
        oldhash = digest(path)
        lines = text.splitlines(keepends=True)
        for first, last, replacement in reversed(spans):
            lines[first-1:last] = [replacement + '\n'] if replacement else ['\n']
        changed = ''.join(lines)
        newtree = ast.parse(changed)
        # The complete AST of every top-level function and class must be preserved.
        bodies = lambda tr: [ast.dump(n, include_attributes=False) for n in tr.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
        if bodies(tree) != bodies(newtree):
            raise RuntimeError(f'Compatibility changed a native function/class: {rel}')
        path.write_text(changed)
        changes.append({'path': rel, 'before_sha256': oldhash, 'after_sha256': digest(path),
                        'function_class_ast_preserved': True,
                        'edits': [{'first_line': f, 'last_line': l, 'replacement': v} for f,l,v in spans]})
    manifest_out.write_text(json.dumps({'profile': 'gpt2-imports-v1', 'changes': changes,
        'scope': 'Import and unused registry isolation only; every top-level native function/class AST is unchanged.'}, indent=2) + '\n')
    return changes
