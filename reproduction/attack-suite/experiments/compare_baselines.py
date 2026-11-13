"""Replay frozen captures with observation ablations and a published baseline.

Only scoring reads canonical tokens, after each prediction file is written.
Results are exploratory reuse of the pilot, not a new held-out confirmation.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacker.evidence import EvidenceLogger, file_sha256
from attacker.experiment import verify_run


def load_observations(bundle, seed, defense, mode, selected_ids, partitions):
    import torch
    from attacker import captures
    folder = bundle / f'seed-{seed}' / defense / mode
    records = captures.scan_captures(str(folder), mode)
    if mode == 'training':
        fwd = [(m, p) for m, p in records if m['phase'] == 'fwd']
        bwd = {(m['step'], m.get('mb_id')): p for m, p in records if m['phase'] == 'bwd'}
        ids = [r['id'] for r in partitions['victim_train']]
        random.Random(seed).shuffle(ids)
    else:
        fwd = [(m, p) for m, p in records if m['phase'] == 'prefill']
        bwd = {}
        ids = [r['id'] for r in partitions['utility_eval']]
    observations = {}
    cursor = 0
    for meta, path in fwd:
        h = captures.load_tensor(path).float()
        if h.ndim == 2:
            h = h.unsqueeze(0)
        g = None
        if mode == 'training':
            key = (meta['step'], meta.get('mb_id'))
            if key not in bwd:
                raise ValueError('missing paired backward observation')
            g = captures.load_tensor(bwd[key]).float()
            if g.ndim == 2:
                g = g.unsqueeze(0)
            if g.shape != h.shape:
                raise ValueError('forward/backward shape mismatch')
        for row in range(len(h)):
            doc = ids[cursor]
            if doc in selected_ids:
                observations[doc] = {'h': h[row], 'g': g[row] if g is not None else None,
                                     'canonical_index': cursor}
            cursor += 1
    if cursor != len(ids) or set(observations) != set(selected_ids):
        raise ValueError('capture/document selection mismatch')
    ordered = [observations[doc] for doc in selected_ids]
    return (torch.stack([r['h'] for r in ordered]),
            torch.stack([r['g'] for r in ordered]) if mode == 'training' else None,
            [r['canonical_index'] for r in ordered], folder / 'canonical.pt')


def score_predictions(predictions, truth_path, indices, document_ids, majority):
    import torch
    # Invoked only after prediction persistence; truth never enters kernels.
    truth = torch.load(truth_path, map_location='cpu', weights_only=True)[indices].long()
    pred = torch.tensor(predictions, dtype=torch.long)
    if pred.shape != truth.shape:
        raise ValueError('prediction/truth shape mismatch')
    correct = (pred == truth).sum(-1)
    shuffled = (pred == truth.roll(1, 0)).sum(-1)
    return {'accuracy': float((pred == truth).float().mean()),
            'exact_sequence_match': float((pred == truth).all(-1).float().mean()),
            'majority_accuracy': float((truth == majority).float().mean()),
            'shuffled_pair_accuracy': float((pred == truth.roll(1, 0)).float().mean()),
            'n_sequences': len(pred), 'n_tokens': pred.numel(),
            'per_sequence': [{'document_id': doc, 'correct': int(c), 'shuffled_correct': int(s),
                              'n_tokens': truth.shape[1]} for doc, c, s in zip(document_ids, correct, shuffled)]}


def expected_arms(protocol, mode):
    names = ['ridge_forward']
    if mode == 'training':
        names += ['direction_backward']
    for k in protocol['candidate_counts']:
        names += [f'forward_beam_k{k}']
        if mode == 'training':
            names += [f'joint_union_k{k}', f'legacy_joint_k{k}']
    names += [f'sipit_steps{s}' for s in protocol['sipit_steps']]
    return names


def run(args):
    import torch
    from attacker.attacks import seam_inversion as si
    from attacker.reconstruction_baselines import (MeteredHead, backward_scores, beam_reconstruct,
                                                   candidate_ids, forward_scores, sipit_reconstruct)
    protocol = json.loads(args.protocol.read_text())
    depth = args.depth
    if depth not in protocol['depths']:
        raise ValueError('depth not in protocol')
    expected_anchor = protocol['source_manifest_sha256'][str(depth)]
    ok, problems = verify_run(args.bundle, expected_anchor)
    if not ok:
        raise ValueError(f'source verification failed: {problems}')
    source = json.loads((args.bundle / 'experiment.json').read_text())
    if int(source['config']['split']) != depth:
        raise ValueError('source depth mismatch')
    if args.output.exists():
        raise ValueError('use a new output directory; prior runs are immutable')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    artifact = args.output / 'comparison.json'
    result = {'schema': 'llm_attacker.baseline_comparison.v1', 'status': 'RUNNING',
              'protocol_sha256': file_sha256(args.protocol), 'depth': depth,
              'source_manifest_sha256': expected_anchor, 'source_experiment_sha256': file_sha256(args.bundle / 'experiment.json'),
              'protocol': protocol, 'results': [], 'setup': [],
              'device': torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
              'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda}}
    started = time.perf_counter()
    with EvidenceLogger(artifact) as evidence:
        try:
            for seed in protocol['seeds']:
                torch.manual_seed(seed)
                begin = time.perf_counter()
                split, embedding = si._load_head_pt(str(args.bundle / f'public-head-{seed}.pt'), args.device)
                split.model.requires_grad_(False)
                embedding = embedding.detach().float()
                head = MeteredHead(split)
                calibration = torch.load(args.bundle / 'calibration.pt', map_location=args.device, weights_only=True)
                majority = si.majority_control_token(calibration.tolist())
                # Warm-up only on public data, outside measurements.
                with torch.no_grad():
                    head(ids=calibration[:1])
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                loading_seconds = time.perf_counter() - begin
                for defense in protocol['defenses']:
                    with head.measure() as fit_work, torch.no_grad():
                        h_cal = head(ids=calibration).reshape(-1, embedding.shape[-1])
                        e_cal = embedding[calibration.flatten()]
                        if defense == 'gaussian':
                            h_cal, e_cal = si._noise_augmented_rows(h_cal, e_cal, protocol['gaussian_sigma'], 8, seed + 911)
                        decoder, _ = si._ridge_fit(h_cal, e_cal, 0)
                    setup_id = f'{seed}-{defense}'
                    result['setup'].append({'id': setup_id, 'model_load_and_warmup_seconds': loading_seconds,
                                            'public_calibration': asdict(fit_work), 'calibration_tokens': calibration.numel()})
                    for mode in protocol['modes']:
                        docs = protocol['selected_document_ids'][mode]
                        h, g, indices, truth_path = load_observations(args.bundle, seed, defense, mode, docs, source['partitions'])
                        h = h.to(args.device)
                        g = g.to(args.device) if g is not None else None
                        arms = expected_arms(protocol, mode)
                        random.Random(seed + depth * 100 + (1 if mode == 'inference' else 0)).shuffle(arms)
                        for arm in arms:
                            torch.manual_seed(seed)
                            with head.measure() as work:
                                with torch.no_grad():
                                    fwd = forward_scores(h, embedding, decoder, head) if arm.startswith(('ridge', 'forward', 'joint', 'legacy')) else None
                                    bwd = backward_scores(g, embedding, head, majority) if arm.startswith(('direction', 'joint', 'legacy')) else None
                                if arm == 'ridge_forward':
                                    pred, diagnostics = fwd.argmax(-1), {}
                                elif arm == 'direction_backward':
                                    pred, diagnostics = bwd.argmax(-1), {'first_position': 'public calibration majority'}
                                elif 'beam_k' in arm or 'union_k' in arm or 'legacy_joint_k' in arm:
                                    count = int(arm.rsplit('k', 1)[1])
                                    variant = 'forward' if arm.startswith('forward') else ('legacy_joint' if arm.startswith('legacy') else 'joint')
                                    candidates = candidate_ids(fwd, bwd, count, variant)
                                    pred, diagnostics = beam_reconstruct(head, h, candidates, protocol['beam_width'])
                                    diagnostics['candidate_ids'] = candidates.cpu().tolist()
                                else:
                                    pred, diagnostics = sipit_reconstruct(head, h, embedding,
                                        steps=int(arm.rsplit('steps', 1)[1]), seed=seed, step_size=protocol['sipit_step_size'])
                                predictions = pred.detach().cpu().tolist()
                                del pred, fwd, bwd
                            name = f'{seed}-{defense}-{mode}-{arm}'
                            prediction_path = args.output / f'{name}.predictions.json'
                            prediction_path.write_text(json.dumps({'document_ids': docs, 'predictions': predictions,
                                                                   'diagnostics': diagnostics}, allow_nan=False))
                            scoring = score_predictions(predictions, truth_path, indices, docs, majority)
                            access = ['public prefix weights', 'public embeddings', 'public calibration', 'sequence length']
                            if not arm.startswith('direction'):
                                access += ['captured activations']
                            if arm.startswith(('direction', 'joint', 'legacy')):
                                access += ['captured seam gradients']
                            cell = {'seed': seed, 'defense': defense, 'mode': mode, 'arm': arm, 'setup_id': setup_id,
                                    **scoring, 'work': asdict(work), 'access': access,
                                    'uses_ridge_calibration': arm.startswith(('ridge', 'forward', 'joint', 'legacy')),
                                    'prediction_file': prediction_path.name, 'prediction_sha256': file_sha256(prediction_path)}
                            result['results'].append(cell)
                            evidence.event('cell_complete', {k:v for k,v in cell.items() if k != 'per_sequence'})
                            artifact.write_text(json.dumps(result, indent=2, allow_nan=False))
                            print(f'{depth=} {seed=} {defense} {mode} {arm}: {scoring["accuracy"]:.4f}, {work.wall_seconds:.2f}s', flush=True)
                        del h, g
                del split, embedding, head, calibration, decoder
            expected = len(protocol['seeds']) * len(protocol['defenses']) * sum(len(expected_arms(protocol, mode)) for mode in protocol['modes'])
            if len(result['results']) != expected:
                raise ValueError('incomplete comparison matrix')
            result['status'] = 'COMPLETE'
        except BaseException as error:
            result['status'] = 'FAILED'
            result['error'] = f'{type(error).__name__}: {error}'
            evidence.event('failure', result['error'])
            raise
        finally:
            result['wall_seconds'] = time.perf_counter() - started
            artifact.write_text(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('experiments/baseline_protocol.json'))
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
