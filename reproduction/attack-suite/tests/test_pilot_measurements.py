import json
import pytest

torch = pytest.importorskip('torch')
from attacker import splitbench as sb
from experiments.run_defended_pilot import _tiny_model
from experiments.pilot_measurements import utility, representation_bundle


def test_replica_input_capture_and_final_state_utility(tmp_path):
    torch.set_num_threads(1)
    model = _tiny_model(11)
    split = sb.build_split(model, sb.SplitPlan.parse('3', n_layers=8))
    cfg = sb.BenchConfig(split='3', defense='replica-median', steps=2,
                        seq_len=8, batch_size=1, capture_dir=str(tmp_path),
                        save_canonical=True, device='cpu').validate()
    batches = [torch.randint(0,64,(1,8)) for _ in range(4)]
    last = {}
    original = split.head_forward
    def head(ids):
        value = original(ids); last['h'] = value.detach().clone(); return value
    split.head_forward = head
    writer = sb.CaptureWriter(str(tmp_path), 'training', 'test')
    write = writer.write_training
    def checked(tensor, **kwargs):
        if kwargs['phase'] == 'fwd':
            assert torch.equal(tensor, last['h'])
        return write(tensor, **kwargs)
    writer.write_training = checked
    stages = []
    def measure(stage, current, config, state):
        stages.append(stage)
        rng = torch.random.get_rng_state().clone()
        journal = len(state['replica'].journal)
        modes = [m.training for m in state['replica_models']]
        result = utility(current, config, state, batches[2:], batches[:2])
        assert result['documents'] == 2 and result['tokens'] == 14
        assert result['partition'] == 'utility_eval'
        assert torch.equal(rng, torch.random.get_rng_state())
        assert len(state['replica'].journal) == journal
        assert [m.training for m in state['replica_models']] == modes
    try:
        sb.finetune(split, cfg, batches[:2], writer, measurement=measure, capture_replica_input=True)
    finally:
        writer.close()
    assert stages == ['before','after']


def test_legacy_pilot_is_rejected_before_comparison(tmp_path):
    from experiments.analyze_defended_pilot import load_cells
    (tmp_path/'cell.json').write_text(json.dumps({'status':'COMPLETE'}))
    cells, failures = load_cells(tmp_path, {'cells':[{'path':'.'}]})
    assert not cells and 'corrected rerun required' in failures[0]['reason']


def test_auxiliary_bundle_rejects_repeated_source_documents():
    from types import SimpleNamespace
    split = sb.build_split(_tiny_model(11), sb.SplitPlan.parse('3', n_layers=8))
    rows = [torch.randint(0,64,(1,8)) for _ in range(8)]
    state={'tb': SimpleNamespace(encode=lambda h:h[...,:4])}
    encoded={'calibration':rows[:4], 'victim_train':rows[4:]}
    ids={'calibration':['a','b','c','d'], 'victim_train':['e','f','g','h']}
    bundle=representation_bundle(split,state,encoded,ids)
    assert len(bundle['calibration'])==4
    ids['victim_train'][0]='a'
    with pytest.raises(ValueError): representation_bundle(split,state,encoded,ids)
    ids['victim_train'][0]='e';encoded['victim_train'][0]=rows[0]
    with pytest.raises(ValueError,match='duplicate token'): representation_bundle(split,state,encoded,ids)
