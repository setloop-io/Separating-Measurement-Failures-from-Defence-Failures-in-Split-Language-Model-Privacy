"""Read-only corpus admission audit and matched upstream mechanical diagnostic.

This is not a recovery experiment or a substitute for the real-data C gate.
Upstream methods are extracted unchanged from the pinned git object, avoiding
the historical checkout's local patches and optional simulator dependencies.
"""
import argparse
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PIN = 'c376590227f9718077765824af4088665bb78687'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def source(repo, name):
    return subprocess.check_output(['git', '-C', str(repo), 'show', PIN + ':' + name], text=True)


def extract(text, name, env):
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError('Ambiguous upstream function: ' + name)
    node = nodes[0]
    node.returns = None
    for arg in node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 '<pinned-upstream:' + name + '>', 'exec'), env)
    return env[name]


def inventory(args):
    import pyarrow as pa
    from experiments.prepare_pilot import articles
    partitions = {}
    for split in ('test', 'train', 'validation'):
        path = args.arrow_dir / f'wikitext-{split}.arrow'
        rows = pa.ipc.open_stream(path).read_all()['text'].to_pylist()
        docs = [{'title': title, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                 'normalised_sha256': hashlib.sha256(' '.join(text.split()).encode()).hexdigest()}
                for title, text in articles(rows)]
        partitions[split] = {'path': str(path), 'sha256': digest(path), 'rows': len(rows),
                             'articles': len(docs), 'documents': docs}
    count = len({d['normalised_sha256'] for d in partitions['test']['documents']})
    save(args.output, {'schema': 'bisr.corpus_admission_audit.v1',
        'dataset': 'Salesforce/wikitext', 'configuration': 'wikitext-2-raw-v1',
        'revision': 'b08601e04326c79dfdd32d625aee71d232d685c3',
        'script_sha256': digest(__file__), 'partitions': partitions,
        'required_documents': 256, 'unique_test_articles_before_exclusions': count,
        'status': 'BLOCKED_INSUFFICIENT_DOCUMENTS' if count < 256 else 'EXCLUSIONS_REQUIRED',
        'historical_exclusions_completed': False, 'launch_admitted': False,
        'note': 'Source inventory only; no eligible selection, windows or checkpoints frozen.'})


def mechanical(args):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from attacker.bisr_fidelity import granted_split
    from attacker.attacks.bisr_baseline import (_bre_gradient_match, _bre_smashed_match,
                                             _bre_decode, _embed_forward)
    torch.set_num_threads(1)
    torch.manual_seed(31415)
    torch.use_deterministic_algorithms(True)
    config = GPT2Config(vocab_size=19, n_positions=8, n_embd=12, n_head=3,
                        n_layer=3, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0,
                        bos_token_id=0, eos_token_id=0)
    config._attn_implementation = 'eager'
    model = GPT2LMHeadModel(config).eval()
    split = granted_split(model, 1)
    # Scorer/capture preparation; neither algorithm is passed these token IDs.
    ids = torch.tensor([[1, 2, 3, 4]])
    h = split.head_forward(ids).detach().requires_grad_(True)
    logits = split.rest_forward(h)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 19), ids[:, 1:].reshape(-1))
    g, = torch.autograd.grad(loss, h)
    scores = torch.randn(1, 4, 19)
    dlg = source(args.upstream, 'sfl/model/attacker/dlg_attacker.py')
    utils = source(args.upstream, 'sfl/utils/model.py')
    sma = source(args.upstream, 'sfl/model/attacker/sma_attacker.py')
    env = {'torch': torch, 'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
           'ParamRestored': lambda **kw: nullcontext()}
    extract(utils, 'calc_shifted_loss_logits', env)
    extract(dlg, '_extract_args_from_inters', env)
    gma_loss = extract(dlg, '_gma_loss', env)
    # Extract TAGAttacker.attack only; no framework setup or parameter replacement.
    cls = next(n for n in ast.parse(dlg).body if isinstance(n, ast.ClassDef) and n.name == 'TAGAttacker')
    attack = extract(ast.unparse(cls), 'attack', env)

    class Top(torch.nn.Module):
        def forward(self, x, **kw):
            return split.rest_forward(x)

    obj = SimpleNamespace(top_mocker=Top(), model_type='decoder', pk=None,
                          target_model_config=config)
    obj._gma_loss = lambda *a, **kw: gma_loss(obj, *a, **kw)
    optimizer = torch.optim.AdamW
    traces = []

    class TracedAdamW(optimizer):
        def step(self, closure=None):
            p = self.param_groups[0]['params'][0]
            trace = {'before': p.detach().clone(), 'gradient': p.grad.detach().clone(),
                     'eps': self.param_groups[0]['eps'],
                     'weight_decay': self.param_groups[0]['weight_decay']}
            result = super().step(closure)
            trace['after'] = p.detach().clone()
            traces.append(trace)
            return result

    torch.optim.AdamW = TracedAdamW
    try:
        ref = attack(obj, SimpleNamespace(model_name='gpt2'),
            SimpleNamespace(softmax=True, init_temp=1.2, epochs=1, lr=.09, beta=.85),
            SimpleNamespace(device='cpu'), None, SimpleNamespace(parameter_keeper=None),
            None, None, SimpleNamespace(fx=h.detach(), grad=g.detach()), {}, init=scores)
        candidate, info = _bre_gradient_match(split, h.detach(), g.detach(), scores,
            iters=1, lr=.09, beta=.85, tau=1.2, opt_batch=1,
            upstream=args.profile == 'upstream')
    finally:
        torch.optim.AdamW = optimizer
    upstream, local = traces
    checks = {}
    for field in ('before', 'gradient', 'after'):
        a = local[field]
        b = upstream[field] if args.profile == 'upstream' else upstream[field][:, 1:]
        checks[field] = {'max_abs_difference': float((a-b).abs().max()),
                        'within_tolerance': bool(torch.all((a-b).abs() <= 1e-6 + 1e-5*b.abs()))}
    # Match SM on the same frozen head. Suppress only progress/metric display;
    # upstream arithmetic, gradient path and optimiser execute unchanged.
    class Holder:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def change_config(self): pass

    class Progress:
        def __init__(self, *a, **kw): pass
        def update(self, *a): pass
        def set_description(self, *a): pass

    class Bottom:
        device = 'cpu'
        fl_config = SimpleNamespace()
        def __init__(self): self.config = config
        def __call__(self, inputs_embeds): return _embed_forward(split, inputs_embeds)

    env.update({'get_embedding_layer': lambda llm: split.wte,
                'ChatGLMForConditionalGenerationSplit': type('UnusedChatGLM', (), {}),
                'FLConfigHolder': Holder, 'tqdm': Progress,
                'evaluate_attacker_rouge': lambda *a: ({'rouge-l': {'f': 0}}, None, None)})
    rec = extract(sma, '_rec_text', env)
    sm_attack = extract(sma, 'attack', env)
    sm_obj = SimpleNamespace(pk=None)
    sm_obj._rec_text = lambda *a: rec(sm_obj, *a)
    traces.clear()
    torch.optim.AdamW = TracedAdamW
    try:
        sm_ref = sm_attack(sm_obj, None,
            SimpleNamespace(at='b2tr', init=True, epochs=1, lr=.01, wd=.01, cosine_loss=True),
            Bottom(), None, SimpleNamespace(parameter_keeper=None), None,
            SimpleNamespace(fx=h.detach()), None, {}, init=scores)
        sm_candidate, _ = _bre_smashed_match(split, split.wte.weight, h.detach(), scores.argmax(-1),
            iters=1, lr=.01, wd=.01, opt_batch=1, upstream=args.profile == 'upstream')
    finally:
        torch.optim.AdamW = optimizer
    sm_upstream, sm_local = traces
    for field in ('before', 'gradient', 'after'):
        a, b = sm_local[field], sm_upstream[field]
        checks['sm_' + field] = {'max_abs_difference': float((a-b).abs().max()),
                                'within_tolerance': bool(torch.all((a-b).abs() <= 1e-6 + 1e-5*b.abs()))}
    checks['gm_tokens'] = {'within_tolerance': bool(torch.equal(ref.detach().argmax(-1), candidate))}
    checks['sm_tokens'] = {'within_tolerance': bool(torch.equal(sm_ref.detach().argmax(-1), sm_candidate))}
    # Execute upstream decoding verbatim on a counterexample with unequal norms.
    emb = torch.nn.Embedding.from_pretrained(torch.tensor([[1., 0.], [10., 0.], [0., 1.]]))
    env.update({'get_embedding_layer': lambda llm: emb,
                'ChatGLMForConditionalGenerationSplit': type('UnusedChatGLM', (), {})})
    rec = extract(sma, '_rec_text', env)
    continuous = torch.tensor([[[1., 0.]]])
    upstream_decode = rec(None, SimpleNamespace(config=SimpleNamespace(vocab_size=3), device='cpu'), continuous).argmax(-1)
    candidate_decode = _bre_decode(continuous, emb.weight, upstream=args.profile == 'upstream')
    tensors = args.output.with_suffix('.pt')
    if tensors.exists():
        raise FileExistsError(tensors)
    torch.save({'model': model.state_dict(), 'h': h.detach(), 'g': g.detach(),
                'sip_logits': scores, 'upstream': upstream, 'candidate': local,
                'sm_upstream': sm_upstream, 'sm_candidate': sm_local,
                'upstream_tokens': ref.detach().argmax(-1), 'candidate_tokens': candidate,
                'decode_embedding_table': emb.weight.detach(), 'decode_input': continuous}, tensors)
    save(args.output, {'schema': 'bisr.upstream_mechanical_diagnostic.v1',
        'scope': 'CPU FP32 one-step GM/SM synthetic fixture, shared public models and observations; not full C',
        'profile': args.profile,
        'upstream_commit': PIN, 'source_hashes': {k: hashlib.sha256(v.encode()).hexdigest()
           for k,v in [('dlg_attacker.py',dlg),('sma_attacker.py',sma),('model.py',utils)]},
        'script_sha256': digest(__file__), 'torch_version': torch.__version__,
        'candidate_sha256': digest(Path(__file__).resolve().parents[1]/'attacker/attacks/bisr_baseline.py'),
        'tensor_file': str(tensors), 'tensor_sha256': digest(tensors), 'checks': checks,
        'optimizers': {k: {f:v[f] for f in ('eps','weight_decay')}
                       for k,v in [('upstream',upstream),('candidate',local)]},
        'decode_counterexample': {'upstream': upstream_decode.tolist(), 'candidate': candidate_decode.tolist()},
        'status': 'CONFORMANCE_FAILED' if not all(c['within_tolerance'] for c in checks.values())
                  or not torch.equal(upstream_decode,candidate_decode) else 'PARTIAL_CHECKS_PASS',
        'launch_admitted': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inventory', 'mechanical'])
    parser.add_argument('--arrow-dir', type=Path)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--profile', choices=['legacy', 'upstream'], default='legacy')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    (inventory if args.mode == 'inventory' else mechanical)(args)
