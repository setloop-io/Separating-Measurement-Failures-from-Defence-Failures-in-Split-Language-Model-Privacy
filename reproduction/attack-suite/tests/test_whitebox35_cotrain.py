"""Final victim/encoder identity and inference-only observation regression."""
import json
from types import SimpleNamespace
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('transformers')
from experiments.whitebox35 import cotrain
from experiments.run_defended_pilot import _tiny_model
from attacker import captures
from attacker.attacks import seam_inversion as si
from attacker.trained_defenses import TrainedBottleneck


@pytest.fixture(scope='module')
def finished(tmp_path_factory):
    torch.set_num_threads(1)
    root=tmp_path_factory.mktemp('cotrain')/'fresh'
    args=cotrain.build_parser().parse_args(['--output',str(root),'--depth','3','--seed','11','--dim','4',
        '--quick','--seq-len','8','--train-steps','2','--tb-minimax-steps','2','--tb-cotrain-interval','1',
        '--tb-hidden','8','--tb-probe-hidden','8','--attack-sequences','2','--sipit-steps','2'])
    generator=torch.Generator().manual_seed(77)
    encoded={k:[torch.randint(0,64,(1,8),generator=generator) for _ in range(3)]
             for k in ['calibration','victim_train','utility_eval']}
    base=_tiny_model(11)
    report=cotrain.run_cell(args,base,encoded,root)
    return root,args,base,encoded,report


def test_final_encoder_and_victim_reproduce_every_inference_wire(finished):
    root,args,base,encoded,report=finished
    split,_=si._load_head_pt(str(root/'final-victim-head.pt'),'cpu')
    defense=TrainedBottleneck.load(root/'final-defense')
    records=list(captures.scan_captures(str(root/'inference-captures'),'inference'))
    assert len(records)==2 and all(m['phase']=='prefill' for m,_ in records)
    with torch.no_grad():
        for (_,path),ids in zip(records,encoded['utility_eval']):
            assert torch.equal(captures.load_tensor(path),defense.apply(split.head_forward(ids)).cpu().half())
    final=torch.load(root/'final-victim.pt',weights_only=True)
    assert any(not torch.equal(final['state_dict'][k],v) for k,v in base.state_dict().items())
    head=torch.load(root/'final-victim-head.pt',weights_only=True)
    assert all(torch.equal(v,final['state_dict'][k]) for k,v in head['state_dict'].items())


def test_observation_is_immutable_and_separate_from_historical_training(finished):
    root,args,base,encoded,report=finished
    entry=report['entry']
    assert entry['state_before_sha256']==entry['state_after_sha256']
    assert entry['optimizer_updates_during_observation']==0
    assert entry['parameter_gradients_unchanged']
    assert report['cotrain_updates']==2
    assert report['historical_training_cells_evaluated'] is False
    assert report['attack']['final_victim_head_granted']
    assert (root/'training-captures/canonical.pt').exists()
    from attacker.evidence import verify_evidence
    assert verify_evidence(root/'cell.json.evidence.json')[0]
    with pytest.raises(ValueError,match='fresh'):
        cotrain.run_cell(args,base,encoded,root)


def test_corrected_objective_has_finite_nonzero_token_embedding_gradient(finished):
    root,args,base,encoded,report=finished
    split,embedding=si._load_head_pt(str(root/'final-victim-head.pt'),'cpu')
    split.model.requires_grad_(False)
    defense=TrainedBottleneck.load(root/'final-defense')
    from attacker.reconstruction_baselines import MeteredHead
    head=MeteredHead(split)
    variable=embedding[encoded['utility_eval'][0]].detach().clone().requires_grad_(True)
    target=defense.encode(head(ids=encoded['utility_eval'][1])).detach()
    loss=(defense.encode(head(embeddings=variable))-target).square().mean()
    loss.backward()
    assert torch.isfinite(variable.grad).all() and variable.grad.abs().max()>0
    assert report['attack']['work']['head_backward_calls']==2


def test_adapter_rejects_training_phase_entry(finished):
    root,args,base,encoded,report=finished
    entry=dict(report['entry'],capture_mode='training')
    with pytest.raises(ValueError,match='inference'):
        cotrain.adaptive_final_inference(args,root,entry,{'attack_sequences':2,'sipit_steps':2})


def test_adapter_rejects_changed_checkpoint_before_optimising(finished):
    root,args,base,encoded,report=finished
    path=root/'final-victim-head.pt'
    original=path.read_bytes()
    try:
        path.write_bytes(original+b'changed')
        with pytest.raises(ValueError,match='hash mismatch'):
            cotrain.adaptive_final_inference(args,root,report['entry'],{'attack_sequences':2,'sipit_steps':2})
    finally:
        path.write_bytes(original)
