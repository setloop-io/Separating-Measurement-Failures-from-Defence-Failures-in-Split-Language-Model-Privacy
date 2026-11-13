"""Acceptance checks for the corrected P6 measurement contract."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments.run_p6_corrected import digest, read_inputs, read_journal, order, heldout_observations, verify_cell


def fixture_data():
    return {p:[{'id':p,'source_document_id':p,'tokens':[1,2,3,4]}]
            for p in ('calibration','train','evaluation')}


def test_input_binding_and_partition_separation(tmp_path):
    p=tmp_path/'input.json';data=fixture_data();p.write_text(json.dumps(data))
    protocol={'input_sha256':digest(p),'partition_counts':dict.fromkeys(data,1),'seq_len':4}
    assert read_inputs(protocol,p)==data
    p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='hash'):read_inputs(protocol,p)
    data['evaluation'][0]['source_document_id']='train';p.write_text(json.dumps(data));protocol['input_sha256']=digest(p)
    with pytest.raises(ValueError,match='Duplicate'):read_inputs(protocol,p)


def test_order_exact_matrix_and_determinism():
    protocol={'starts':[0,1,2,3],'defenses':['D0','D1'],'victim_seeds':[11,22,33,44,55,66,77,88],'order_seed':41778}
    x=order(protocol,2)
    assert len(x)==len(set(x))==64
    assert x==order(protocol,2)
    assert [c[1:] for c in x]!=[c[1:] for c in order(protocol,4)]


def test_malformed_journal_is_rejected(tmp_path):
    p=tmp_path/'journal';p.write_text('{"rc":0}\ntruncated')
    with pytest.raises(json.JSONDecodeError):read_journal(p)


def test_empty_summary_is_not_a_verified_cell(tmp_path):
    (tmp_path/'cell.json').write_text('{"summary":[{}]}')
    with pytest.raises(ValueError,match='not complete'):verify_cell(tmp_path)


def tiny_split():
    torch=pytest.importorskip('torch')
    tf=pytest.importorskip('transformers')
    from attacker import splitbench as sb
    torch.manual_seed(91)
    model=tf.GPT2LMHeadModel(tf.GPT2Config(n_layer=2,n_head=2,n_embd=16,vocab_size=31,n_positions=16))
    return torch,sb,sb.build_split(model,sb.SplitPlan.parse('1',n_layers=2))


def test_heldout_observation_preserves_weights_rng_modes_and_records_rows(tmp_path):
    torch,sb,split=tiny_split()
    cfg=sb.BenchConfig(model='unused',split='1',defense='quant',quant_bits=4,quant_grad=True,grad_noise=.1,seed=11)
    cfg.validate();split.model.train()
    state=sb.make_defense_state(cfg,split,calibration=[],out_dir=tmp_path)
    before={k:v.clone() for k,v in split.model.state_dict().items()}
    rng=torch.get_rng_state().clone()
    rows=[{'id':'a','source_document_id':'a','tokens':[1,2,3,4]}, {'id':'b','source_document_id':'b','tokens':[2,3,4,5]}]
    results=heldout_observations(split,cfg,state,rows,tmp_path/'captures')
    assert split.model.training and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,split.model.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in split.model.parameters())
    assert [r['document_id'] for r in results]==['a','b']
    assert len(list((tmp_path/'captures').glob('wire_*.pt')))==4
    canonical=torch.load(tmp_path/'captures/canonical.pt',weights_only=True)
    assert canonical.tolist()==[r['tokens'] for r in rows]
    assert all(r['loss']>0 and r['gradient_norm']>0 for r in results)


def test_pinned_calibration_cli_roundtrip(tmp_path):
    torch=pytest.importorskip('torch')
    from attacker.attacks.seam_inversion import build_parser,_calibration_ids
    p=tmp_path/'public.pt';torch.save(torch.tensor([[1,2,3,4],[2,3,4,5]]),p)
    args=build_parser().parse_args(['--calibration-pt',str(p),'--calibration-seqs','2'])
    assert _calibration_ids(args,31,4,'cpu').tolist()==[[1,2,3,4],[2,3,4,5]]
    args.calibration_file='ambiguous.txt'
    with pytest.raises(ValueError,match='exclusive'):_calibration_ids(args,31,4,'cpu')


def test_variance_test_does_not_infer_start_from_span_only():
    import numpy as np
    from experiments.analyze_p6_corrected import variance_test,prediction_comparison
    x=np.broadcast_to(np.array([2.,4.,6.])[None,:,None],(8,3,4)).copy()
    result=variance_test(x,99,11,.0125)
    assert result['partial_r2']==0 and not result['start_matters']
    x += np.arange(4)[None,None,:]*10
    result=variance_test(x,199,11,.0125)
    assert result['start_matters']
    prediction=prediction_comparison(x,[11,22,33,44,55,66,77,88])
    assert prediction['ten_percent_improvement']


def test_complete_cell_tiny_model_and_tamper_rejection(tmp_path,monkeypatch):
    import random
    torch=pytest.importorskip('torch');tf=pytest.importorskip('transformers')
    import huggingface_hub
    from attacker import splitbench as sb
    from experiments.run_p6_corrected import cell
    model_dir=tmp_path/'model'
    model=tf.GPT2LMHeadModel(tf.GPT2Config(n_layer=2,n_head=2,n_embd=16,vocab_size=31,n_positions=32))
    model.save_pretrained(model_dir)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda *a,**kw:str(model_dir))
    monkeypatch.setattr(sb,'_load_model_and_tokenizer',lambda cfg:(tf.AutoModelForCausalLM.from_pretrained(model_dir),None))
    rng=random.Random(42);data={}
    for part,n in [('calibration',32),('train',128),('evaluation',256)]:
        data[part]=[{'id':f'{part}-{i}','source_document_id':f'{part}-{i}',
                     'tokens':[rng.randrange(1,31) for _ in range(32)]} for i in range(n)]
    inp=tmp_path/'input.json';inp.write_text(json.dumps(data))
    protocol={'status':'frozen','implementation_sha256':{},'input':str(inp),'input_sha256':digest(inp),
      'partition_counts':{'calibration':32,'train':128,'evaluation':256},'seq_len':32,'train_steps':1,
      'spans':[2],'starts':[0],'defenses':['D0'],'victim_seeds':[11],'lr':5e-5,
      'model':'unused','model_revision':'test-only','attack_seed':17,'property_tokens':[1,2,3],
      'attack_timeout_s':120}
    pp=tmp_path/'protocol.json';pp.write_text(json.dumps(protocol))
    args=SimpleNamespace(threads=1,protocol=str(pp),workdir=str(tmp_path/'cell'),span=2,start=0,
                         defense='D0',seed=11,kind='trained',device='cpu')
    cell(args)
    report=verify_cell(tmp_path/'cell',digest(pp));assert report['status']=='complete'
    assert report['final_utility']['documents']==256
    with (tmp_path/'cell/measurements.json').open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='hash mismatch'):verify_cell(tmp_path/'cell',digest(pp))
