"""Engineering conformance checks; these tests do not qualify a real campaign."""
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
from attacker import bisr_fidelity as bf


def test_equivalence_does_not_turn_non_significance_into_agreement():
    assert bf.equivalence([0, 0, 0])['status'] == 'EQUIVALENT'
    assert bf.equivalence([-.1, 0, .1])['status'] == 'INCONCLUSIVE'
    assert bf.equivalence([.01] * 3)['status'] == 'INCONCLUSIVE'
    assert bf.equivalence([.02] * 3)['status'] == 'NON_EQUIVALENT'
    for bad in ([0, 0], [0, 0, float('nan')]):
        with pytest.raises(ValueError):
            bf.equivalence(bad)


def test_all_eight_endpoints_and_complete_pairing_required():
    scores = {str(seed): {a: [{'sample_id': str(i), 'positional_accuracy': .5,
                             'rouge_l_f1': .5} for i in range(256)]
                         for a in ('sip', 'b', 'f', 'bf')}
              for seed in (31415, 27182, 16180)}
    out = bf.compare_scores(scores, scores)
    assert len(out['endpoints']) == 8 and out['statistical_status'] == 'EQUIVALENT'
    assert out['launch_admitted'] is False
    changed = deepcopy(scores)
    for seed in changed:
        for r in changed[seed]['f']:
            r['rouge_l_f1'] += .02
    assert bf.compare_scores(changed, scores)['statistical_status'] == 'NON_EQUIVALENT'
    changed['31415']['sip'].pop()
    with pytest.raises(ValueError):
        bf.compare_scores(changed, scores)


def test_export_is_immutable_and_hash_bound(tmp_path):
    path = tmp_path / 'predictions.json'
    assert bf.write_predictions(path, {'value': [1, 2]}) == bf.sha256(path)
    with pytest.raises(FileExistsError):
        bf.write_predictions(path, {'value': [3]})
    with pytest.raises(ValueError):
        bf.write_predictions(tmp_path / 'bad.json', {'value': float('nan')})


@pytest.fixture
def world():
    torch = pytest.importorskip('torch')
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.set_num_threads(1)
    torch.manual_seed(218)
    config = GPT2Config(vocab_size=19, n_positions=8, n_embd=12, n_head=3,
                        n_layer=3, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)
    config._attn_implementation = 'eager'
    model = GPT2LMHeadModel(config).eval()
    split = bf.granted_split(model, 1)
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    h = split.head_forward(ids).detach().requires_grad_(True)
    logits = split.rest_forward(h)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 19), ids[:, 1:].reshape(-1))
    g, = torch.autograd.grad(loss, h)
    bundle = {'schema': 'bisr.observations.v1', 'sample_ids': ['one', 'two'],
              'h': h.detach(), 'g': g.detach(), 'mask': torch.ones_like(ids),
              'gradient_batch_scale': torch.tensor([2., 2.])}
    return torch, model, split, ids, bundle


def test_gru_matches_independent_gate_equations_and_is_causal(world):
    torch, _, _, _, _ = world
    inv = bf.gru_class()(19, 12, 4, hidden_size=7, dropout=.1).eval()
    x = torch.randn(2, 4, 12, requires_grad=True)
    hidden = torch.zeros(2, 7)
    outputs = []
    # Independent recurrence, not a second invocation of nn.GRU.
    for t in range(4):
        xi = torch.nn.functional.linear(x[:, t], inv.gru.weight_ih_l0, inv.gru.bias_ih_l0)
        hh = torch.nn.functional.linear(hidden, inv.gru.weight_hh_l0, inv.gru.bias_hh_l0)
        ir, iz, inn = xi.chunk(3, -1); hr, hz, hn = hh.chunk(3, -1)
        reset = torch.sigmoid(ir + hr); update = torch.sigmoid(iz + hz)
        new = torch.tanh(inn + reset * hn)
        hidden = (1 - update) * new + update * hidden
        outputs.append(inv.output(hidden))
    expected = torch.stack(outputs, 1)
    actual = inv(x)
    bf.numerical_check(actual, expected)
    ga, = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)
    ge, = torch.autograd.grad(expected.square().sum(), x)
    bf.numerical_check(ga, ge)
    future = x.detach().clone(); future[:, 2:] += 10
    torch.testing.assert_close(inv(future)[:, :2], actual[:, :2])
    assert inv.dropout.p == .1 and inv.gru.num_layers == 1 and not inv.gru.bidirectional


def test_gru_epochs_visit_every_row_including_tail(world):
    torch, _, _, ids, bundle = world
    seen = []
    def fetch(indices):
        seen.extend(indices)
        return ids[indices], bundle['h'][indices]
    inv = bf.gru_class()(19, 12, 4, hidden_size=7)
    report = bf.train_gru(inv, fetch, 2, epochs=3, batch_size=3, seed=1)
    assert report['steps'] == 3
    assert all(sorted(seen[i:i+2]) == [0, 1] for i in (0, 2, 4))
    assert report['weight_decay'] == 0


def test_exact_grant_changes_only_bottom_and_rejects_bad_exports(world, tmp_path):
    torch, model, _, ids, _ = world
    from attacker.attacks.seam_inversion import save_head_pt
    victim = deepcopy(model)
    with torch.no_grad():
        victim.transformer.wte.weight.add_(.2)
        victim.transformer.h[0].ln_1.bias.add_(.1)
    path = tmp_path / 'head.pt'
    save_head_pt(path, victim, 1)
    digest = bf.sha256(path)
    exact = bf.granted_split(model, 1, knowledge='exact', head_path=path, head_sha256=digest)
    public = bf.granted_split(model, 1)
    assert not torch.equal(exact.head_forward(ids), public.head_forward(ids))
    wire = public.head_forward(ids)
    torch.testing.assert_close(exact.rest_forward(wire), public.rest_forward(wire), rtol=0, atol=0)
    assert not torch.equal(exact.wte.weight, exact.public.lm_head.weight)
    assert torch.equal(model.lm_head.weight, public.lm_head.weight)
    for kwargs in ({'head_path': path, 'head_sha256': digest},
                   {'knowledge': 'exact'},
                   {'knowledge': 'exact', 'head_path': path, 'head_sha256': '0'*64}):
        with pytest.raises(ValueError):
            bf.granted_split(model, 1, **kwargs)
    blob = torch.load(path, weights_only=True)
    blob['state_dict']['lm_head.weight'] = victim.lm_head.weight.detach()
    torch.save(blob, path)
    with pytest.raises(ValueError, match='all and only'):
        bf.granted_split(model, 1, knowledge='exact', head_path=path, head_sha256=bf.sha256(path))


def test_wire_replay_and_parameter_immutability(world):
    torch, _, split, ids, bundle = world
    before = {k: v.clone() for k, v in split.model.state_dict().items()}
    assert bf.replay_wire(split, ids, bundle['h'], bundle['g'])
    with pytest.raises(ValueError, match='replay'):
        bf.replay_wire(split, ids, bundle['h'], bundle['g'] + .01)
    assert all(torch.equal(before[k], v) for k, v in split.model.state_dict().items())


def test_prediction_boundary_rejects_truth_padding_duplicate_ids(world):
    torch, _, _, _, bundle = world
    for change in ({'true_ids': []}, {'sample_ids': ['same', 'same']},
                   {'mask': torch.zeros_like(bundle['mask'])},
                   {'gradient_batch_scale': torch.tensor([1., float('nan')])}):
        with pytest.raises(ValueError):
            bf.validate_observations({**bundle, **change})


def test_forward_gradient_and_independent_adamw_step(world):
    torch, _, split, ids, bundle = world
    from attacker.attacks.bisr_baseline import _embed_forward, _bre_smashed_match
    embedding = split.wte.weight[ids].detach().clone().requires_grad_(True)
    bf.numerical_check(_embed_forward(split, embedding), split.head_forward(ids))
    target = bundle['h'] + .01
    loss = (1 - torch.nn.functional.cosine_similarity(_embed_forward(split, embedding), target, dim=-1)).mean()
    gradient, = torch.autograd.grad(loss, embedding)
    assert gradient.abs().sum() > 0
    # AdamW's first update has bias-corrected m=g and v=g^2.
    expected = embedding.detach() * (1 - .01*.01) - .01 * gradient / (gradient.abs() + 1e-8)
    actual = embedding.detach().clone().requires_grad_(True)
    opt = torch.optim.AdamW([actual], lr=.01, weight_decay=.01, betas=(.9,.999), eps=1e-8)
    actual.grad = gradient.clone(); opt.step()
    bf.numerical_check(actual, expected)
    # Validate the actual solver's first-step nearest-embedding predictions.
    pred, _ = _bre_smashed_match(split, split.wte.weight, bundle['h'][:1]+.01,
                                ids[:1], iters=1, lr=.01, wd=.01, opt_batch=1)
    single = embedding.detach()[:1].clone().requires_grad_(True)
    ls = (1-torch.nn.functional.cosine_similarity(_embed_forward(split,single), target[:1],dim=-1)).mean()
    grad, = torch.autograd.grad(ls,single)
    upd = single.detach()*(1-.0001)-.01*grad/(grad.abs()+1e-8)
    assert torch.equal(pred, torch.cdist(upd,split.wte.weight).argmin(-1))
    assert all(p.grad is None for p in split.model.parameters())


def test_gm_alignment_and_stage_order_no_beam(world, monkeypatch):
    torch, _, split, _, bundle = world
    from attacker.attacks import bisr_baseline as bb
    inv = bf.gru_class()(19,12,4,hidden_size=7).eval()
    sip = inv(bundle['h']).argmax(-1)
    calls = []
    def gm(split, h, g, scores, **kwargs):
        assert kwargs['upstream'] is True
        calls.append('gm')
        result = scores.argmax(-1).clone(); result[:,1:] = (result[:,1:]+1)%19
        return result, {'gm_final_loss_mean':0.0}
    def sm(split, E, h, initial, **kwargs):
        assert kwargs['upstream'] is True
        calls.append(initial.clone()); return initial.clone(), {'sm_final_loss_mean':0.0}
    monkeypatch.setattr(bb, '_bre_gradient_match', gm)
    monkeypatch.setattr(bb, '_bre_smashed_match', sm)
    args = SimpleNamespace(gm_iters=1,sm_iters=1,gm_lr=.09,gm_beta=.85,gm_tau=1.2,sm_lr=.01,sm_wd=.01,opt_batch=1)
    pred = bf.predict_arms(split, inv, bundle, args)
    assert calls[0] == 'gm' and torch.equal(calls[1], sip)
    assert torch.equal(calls[2], torch.tensor(pred['arms']['b']))
    assert [r[0] for r in pred['arms']['b']] == sip[:,0].tolist()
    assert set(pred['arms']) == {'sip','b','f','bf'}
    assert pred['algorithm_profile'] == 'splitllm-c376590'


def test_actual_gm_matches_independent_single_step_and_shift(world):
    torch, _, split, _, bundle = world
    from attacker.attacks.bisr_baseline import _bre_gradient_match
    scores = torch.randn(1,4,19)
    h = bundle['h'][:1].clone().requires_grad_(True)
    g = bundle['g'][:1]*2
    z = (scores[:,1:]*1.2).clone().requires_grad_(True)
    log_probs = split.rest_forward(h)[:,:-1].log_softmax(-1)
    ce = -(z.div(1.2).softmax(-1)*log_probs).sum(-1).mean()
    dg, = torch.autograd.grad(ce,h,create_graph=True)
    delta = dg-g
    objective = .85*torch.sqrt((delta**2).sum())+.15*delta.abs().sum()
    dz, = torch.autograd.grad(objective,z)
    upd = z.detach()-.09*dz/(dz.abs()+1e-8)
    expected = scores.argmax(-1);expected[:,1:] = upd.argmax(-1)
    actual,_ = _bre_gradient_match(split,bundle['h'][:1],g,scores,iters=1,lr=.09,beta=.85,tau=1.2,opt_batch=1)
    assert torch.equal(actual,expected)


def test_metric_identity_and_label_isolation(world):
    pytest.importorskip('rouge');pytest.importorskip('nltk')
    torch, _, split, ids, bundle = world
    class Tokenizer:
        def decode(self, token, skip_special_tokens=True): return ['red','blue','green'][int(token)%3]
        def tokenize(self, text): return text.split()
    pred = {'schema':'bisr.predictions.v1','sample_ids':['x'],
            'arms':{a:[[0,1,2]] for a in ('sip','b','f','bf')}}
    truth = {'schema':'bisr.truth.v1','sample_ids':['x'],'token_ids':[[0,1,2]]}
    scores = bf.score_predictions(pred,truth,Tokenizer())
    row=scores['sip'][0]
    assert row['positional_accuracy']==1 and row['token_set_f1']==1
    # Independent LCS F1 formula including rouge package's denominator epsilon.
    assert abs(row['rouge_l_f1']-2/(2+1e-8))<1e-12
    from nltk.translate.meteor_score import meteor_score
    assert row['meteor']==meteor_score([['red','blue','green']],['red','blue','green'])
    original=deepcopy(pred)
    truth['token_ids']=[[2,1,0]]
    bf.score_predictions(pred,truth,Tokenizer())
    assert pred==original


def test_real_prediction_core_is_deterministic_and_label_free(world):
    torch, _, split, _, bundle = world
    inv=bf.gru_class()(19,12,4,hidden_size=7).eval()
    args=SimpleNamespace(gm_iters=1,sm_iters=1,gm_lr=.09,gm_beta=.85,gm_tau=1.2,sm_lr=.01,sm_wd=.01,opt_batch=1)
    assert bf.predict_arms(split,inv,bundle,args)==bf.predict_arms(split,inv,bundle,args)


def test_gru_checkpoint_roundtrip_through_existing_cli(world, tmp_path, monkeypatch):
    torch, model, _, ids, bundle = world
    from attacker.attacks import bisr_baseline as bb
    from attacker.attacks import seam_inversion as si
    monkeypatch.setattr(bb, '_load_hf_model', lambda *a: deepcopy(model))
    args = bb.build_parser().parse_args(['--model','fixture','--split','1',
        '--decoder','gru','--train-epochs','1','--aux-seqs','2','--seq-len','4',
        '--batch-size','2','--inverter-pt',str(tmp_path/'inv.pt'),
        '--output',str(tmp_path/'train.json'),'--device','cpu'])
    args.mode='training'
    bb._run_train(args,None)
    checkpoint=torch.load(args.inverter_pt,weights_only=True)
    assert checkpoint['format']==bf.GRU_FORMAT
    assert checkpoint['arch']['hidden_size']==256
    assert checkpoint['train']['epoch_training']['steps']==1
    picked=[{'h':bundle['h'][i],'g':bundle['g'][i],'batch':2,
             'true':ids[i].tolist(),'step':0,'epoch':0} for i in range(2)]
    monkeypatch.setattr(si,'_load_scored_sequences_joint',lambda args:(picked,4,12))
    args.output=str(tmp_path/'attack.json');args.opt_arm='bf'
    args.gm_iters=1;args.sm_iters=1;args.opt_batch=1
    bb._run_attack(args,None)
    record=json.loads((tmp_path/'attack.json').read_text())
    assert record['config']['bre_initialisation']=='raw SIP; beam is separate'
    assert len(record['results'][0]['predictions']['bre'])==2


def test_prediction_process_never_loads_scorer_truth(world, tmp_path):
    import subprocess,sys
    from pathlib import Path
    torch, model, _, _, bundle=world
    def save(name, obj):
        path=tmp_path/name;torch.save(obj,path)
        return {'path':name,'sha256':bf.sha256(path)}
    inv=bf.gru_class()(19,12,4)
    manifest={'schema':'bisr.prediction_manifest.v1',
        'public_model':save('public.pt',{'config':model.config.to_dict(),'state_dict':model.state_dict()}),
        'inverter':save('inverter.pt',{'format':bf.GRU_FORMAT,'arch':{'vocab':19,'seam_dim':12,'seq_len':4,'hidden_size':256,'dropout':.1},'state_dict':inv.state_dict()}),
        'observations':save('wires.pt',bundle),'knowledge':'public','head':None,'boundary':1,
        'bre':{'gm_iters':1,'gm_lr':.09,'gm_beta':.85,'gm_tau':1.2,'sm_iters':1,'sm_lr':.01,'sm_wd':.01,'opt_batch':1}}
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(manifest))
    runner=Path(__file__).resolve().parents[1]/'experiments/bisr_fidelity_predict.py'
    command=[sys.executable,str(runner),'--manifest',str(mp),'--output']
    truth=tmp_path/'scorer-only-truth.json'
    truth.write_text('[1,2,3,4]')
    subprocess.run(command+[str(tmp_path/'first.json')],check=True,capture_output=True)
    truth.write_text('[4,3,2,1]')
    subprocess.run(command+[str(tmp_path/'second.json')],check=True,capture_output=True)
    assert (tmp_path/'first.json').read_bytes()==(tmp_path/'second.json').read_bytes()
    manifest['truth_path']=str(truth);mp.write_text(json.dumps(manifest))
    result=subprocess.run(command+[str(tmp_path/'forbidden.json')],capture_output=True)
    assert result.returncode!=0 and not (tmp_path/'forbidden.json').exists()


def test_unsupported_attention_backend_is_rejected(world):
    _,model,_,_,_=world
    model.config._attn_implementation='sdpa'
    with pytest.raises(ValueError,match='eager'):
        bf.granted_split(model,1)


def test_upstream_profile_preserves_position_zero_and_decode_rule(world):
    torch, _, split, _, bundle = world
    from attacker.attacks.bisr_baseline import _bre_gradient_match, _bre_decode
    scores = torch.randn(1, 4, 19)
    pred, info = _bre_gradient_match(split, bundle['h'][:1], bundle['g'][:1]*2,
        scores, iters=2, lr=.09, beta=.85, tau=1.2, opt_batch=1, upstream=True)
    assert pred[0,0] == scores.argmax(-1)[0,0]
    assert info['algorithm_profile'] == 'splitllm-c376590'
    embeddings = torch.tensor([[[1., 0.]]])
    table = torch.tensor([[1.,0.],[10.,0.],[0.,1.]])
    assert _bre_decode(embeddings, table, upstream=True).item() == 1
    assert _bre_decode(embeddings, table).item() == 0


def test_amended_comparison_requires_frozen_ids_not_observed_count():
    manifest = {'schema':'bisr.frozen_inputs.v2', 'validation_ids':['a','b'],
                'validation_documents':2}
    scores = {str(seed):{arm:[{'sample_id':i,'positional_accuracy':.5,'rouge_l_f1':.5}
                             for i in ['a','b']] for arm in ('sip','b','f','bf')}
              for seed in (31415,27182,16180)}
    assert bf.compare_frozen_scores(scores,scores,manifest)['statistical_status']=='EQUIVALENT'
    altered=deepcopy(scores)
    altered['31415']['sip'][0]['sample_id']='wrong'
    with pytest.raises(ValueError, match='frozen validation identities'):
        bf.compare_frozen_scores(altered,altered,manifest)
