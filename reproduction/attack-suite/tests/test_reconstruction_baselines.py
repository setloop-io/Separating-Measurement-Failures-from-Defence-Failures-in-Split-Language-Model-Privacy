"""Behavioral checks for observation isolation, compute parity and inversion."""
import inspect

import pytest

pytestmark = pytest.mark.slow

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
from transformers import GPT2Config, GPT2LMHeadModel
from attacker.splitbench import SplitPlan, build_split
from attacker.reconstruction_baselines import (MeteredHead, backward_scores, beam_reconstruct,
                                               candidate_ids, sipit_reconstruct)


@pytest.fixture
def head():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=24, n_layer=3, n_head=4,
        n_positions=32, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval().requires_grad_(False)
    return MeteredHead(build_split(model, SplitPlan.parse('2', n_layers=3)))


def test_joint_proposals_preserve_forward_and_fill_overlap():
    fwd = torch.tensor([[[9., 8., 7., 6., 5., 4.], [9., 8., 7., 6., 5., 4.]]])
    bwd = fwd.flip(-1)
    ids = candidate_ids(fwd, bwd, 4, 'joint')
    assert ids[0, 0].tolist() == [0, 1, 2, 3]
    assert ids[0, 1].tolist() == [0, 1, 5, 4]
    overlap = candidate_ids(fwd, fwd, 4, 'joint')
    assert overlap[0, 1].tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        candidate_ids(fwd, None, 4, 'joint')


def test_matched_beams_have_identical_prefix_work(head):
    ids = torch.tensor([[3, 5, 7], [9, 11, 13]])
    h = head(ids=ids).detach()
    scores = torch.randn(2, 3, 64)
    work = []
    for variant in ('forward', 'joint'):
        candidates = candidate_ids(scores, scores.flip(-1), 8, variant)
        with head.measure() as measured:
            pred, _ = beam_reconstruct(head, h, candidates, 2)
        assert pred.shape == ids.shape
        work.append((measured.head_forward_calls, measured.head_forward_sequences,
                     measured.head_forward_token_positions))
    assert work[0] == work[1] == (3, 2 * 8 * 5, 2 * 8 * (1 + 2 * (2 + 3)))


def test_backward_arm_has_no_activation_input_and_no_first_label(head):
    assert 'h' not in inspect.signature(backward_scores).parameters
    g = torch.randn(2, 4, 24)
    first = backward_scores(g, head.split.wte.weight, head, 17)
    second = backward_scores(g * -3, head.split.wte.weight, head, 17)
    assert first[:, 0].argmax(-1).tolist() == second[:, 0].argmax(-1).tolist() == [17, 17]
    assert 'truth' not in inspect.signature(sipit_reconstruct).parameters
    assert 'truth' not in inspect.signature(beam_reconstruct).parameters


def test_sipit_recovers_exact_causal_tokens(head):
    ids = torch.tensor([[3, 17, 45, 2]])
    h = head(ids=ids).detach()
    with head.measure():
        recovered, details = sipit_reconstruct(head, h, head.split.wte.weight.detach(), steps=64, seed=11)
    assert torch.equal(recovered, ids)
    assert all(all(row) for row in details['exact_match_by_position'])


def test_sipit_noisy_observations_have_bounded_finite_fallback(head):
    ids = torch.tensor([[3, 17, 45, 2], [12, 3, 4, 5]])
    h = head(ids=ids).detach() + .03 * torch.randn(2, 4, 24)
    with head.measure() as work:
        recovered, details = sipit_reconstruct(head, h, head.split.wte.weight.detach(), steps=3, seed=11)
    assert recovered.shape == ids.shape
    assert all(count <= 3 for row in details['iterations_by_position'] for count in row)
    assert torch.isfinite(torch.tensor(details['best_discrete_mse_by_position'])).all()
    assert work.head_forward_calls <= 2 * 3 * 4
    assert work.head_backward_calls <= 3 * 4


def test_sipit_batch_does_not_rescale_individual_gradients(head):
    ids = torch.tensor([[3, 17, 45, 2]])
    h = head(ids=ids).detach()
    embedding = head.split.wte.weight.detach()
    one, _ = sipit_reconstruct(head, h, embedding, steps=64, seed=11)
    two, _ = sipit_reconstruct(head, h.expand(2, -1, -1), embedding, steps=64, seed=11)
    assert torch.equal(one[0], two[0]) and torch.equal(two[0], two[1])


def test_beam_can_recover_exact_sequence_if_candidates_cover_it(head):
    ids = torch.tensor([[3, 17, 45, 2]])
    h = head(ids=ids).detach()
    candidates = torch.stack((ids, (ids + 1) % 64), -1)
    recovered, _ = beam_reconstruct(head, h, candidates)
    assert torch.equal(recovered, ids)


def test_capture_selection_never_loads_scoring_truth(monkeypatch, tmp_path):
    from attacker import captures
    from experiments.compare_baselines import load_observations
    records, tensors = [], {}
    for index in range(3):
        for phase in ['fwd', 'bwd']:
            path = tmp_path / f'{index}-{phase}.pt'
            records.append(({'phase': phase, 'step': index, 'mb_id': index}, path))
            tensors[path] = torch.full((1, 4, 8), float(index + (10 if phase == 'bwd' else 0)))
    monkeypatch.setattr(captures, 'scan_captures', lambda *a: records)
    monkeypatch.setattr(captures, 'load_tensor', lambda path: tensors[path])
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('truth opened during observation loading'))
    partitions = {'victim_train': [{'id': f'doc{i}'} for i in range(3)]}
    h, g, indices, _ = load_observations(tmp_path, 11, 'none', 'training', ['doc1', 'doc0'], partitions)
    assert indices == [2, 0]
    assert h[:, 0, 0].tolist() == [2, 0]
    assert g[:, 0, 0].tolist() == [12, 10]
    records.pop()  # remove the paired gradient for the last capture
    with pytest.raises(ValueError, match='missing paired backward'):
        load_observations(tmp_path, 11, 'none', 'training', ['doc1', 'doc0'], partitions)
