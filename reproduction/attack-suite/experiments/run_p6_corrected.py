#!/usr/bin/env python3
"""Prospective P6 follow-up; the historical run_p6_grid.py is immutable.

Run as python -m experiments.run_p6_corrected {cell,grid} --help.
Every fresh cell trains on the pinned train partition and observes the final
victim on the pinned held-out documents, with no evaluation optimizer updates.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_PROTOCOL = ROOT / 'experiments/crossed_span_protocol.p6.v2.json'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def read_inputs(protocol, path):
    if digest(path) != protocol['input_sha256']:
        raise ValueError('Pinned input hash mismatch')
    data = json.loads(Path(path).read_text())
    seen = set()
    for part, count in protocol['partition_counts'].items():
        rows = data[part]
        if len(rows) != count:
            raise ValueError('Partition count mismatch: ' + part)
        for row in rows:
            ident = row['source_document_id']
            if ident in seen:
                raise ValueError('Duplicate source document across partitions')
            seen.add(ident)
            if len(row['tokens']) != protocol['seq_len']:
                raise ValueError('Sequence length mismatch')
    return data


def order(protocol, span):
    cells = [(span, start, defense, seed) for start in protocol['starts']
             for defense in protocol['defenses'] for seed in protocol['victim_seeds']]
    random.Random(protocol['order_seed'] + span).shuffle(cells)
    return cells


def tag(span, start, defense, seed):
    return f'span{span}_b{start}_{defense}_s{seed}'


def read_journal(path):
    if not path.exists():
        return []
    # Invalid/truncated lines are evidence failures, never silently discarded.
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_journal(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def verify_cell(root, protocol_sha=None):
    from attacker.evidence import verify_evidence
    root = Path(root)
    report = json.loads((root / 'cell.json').read_text())
    if report.get('status') != 'complete':
        raise ValueError('Cell is not complete')
    if protocol_sha and report['protocol_sha256'] != protocol_sha:
        raise ValueError('Cell protocol hash mismatch')
    for rel, want in report['files'].items():
        p = root / rel
        if not p.is_file() or digest(p) != want:
            raise ValueError('Cell file hash mismatch: ' + rel)
    for p in sorted(root.rglob('*.evidence.json')):
        ok, problems = verify_evidence(p)
        if not ok:
            raise ValueError(f'Evidence failed: {p}: {problems}')
    return report


def heldout_observations(split, cfg, state, rows, directory):
    """Final trained victim; gradients observed without optimizer updates.

    State and RNG restored on exit. Gradients are computed with respect to
    the detached seam leaf only; this does not accumulate parameter .grad.
    """
    import torch
    import torch.nn.functional as F
    from attacker import splitbench as sb
    from experiments.pilot_measurements import evaluation_state
    from attacker.attacks.label_scoring import _sequence_norms_torch
    directory = Path(directory)
    records, canonical = [], []
    with evaluation_state(split, state):
        torch.manual_seed(cfg.seed + 90317)
        with sb.CaptureWriter(directory, 'training', f'p6-heldout-{cfg.seed}') as writer:
            for i, row in enumerate(rows):
                ids = torch.tensor([row['tokens']], dtype=torch.long, device=split.device)
                with torch.no_grad():
                    hidden = split.head_forward(ids)
                    sent = sb._apply_seam_defense(hidden, cfg, state)
                wire = sent.detach().requires_grad_(True)
                logits = sb._server_rest_forward(split, wire, cfg, state)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                       ids[:, 1:].reshape(-1))
                gradient = torch.autograd.grad(loss, wire)[0]
                observed = sb._apply_grad_defense(gradient, cfg)
                writer.write_training(wire.detach(), 0, 'fwd', i, 0)
                writer.write_training(observed.detach(), 0, 'bwd', i, 0)
                # Norms below use the same fp16 round trip as the persisted wire.
                norm = _sequence_norms_torch(observed.detach().to('cpu', torch.float16))[0]
                records.append({'document_id': row['id'], 'source_document_id': row['source_document_id'],
                                'step': i, 'loss': float(loss.detach()), 'n_tokens': len(row['tokens']) - 1,
                                'gradient_norm': norm})
                canonical.append(ids.cpu())
        torch.save(torch.cat(canonical), directory / 'canonical.pt')
    return records


def utility_rows(split, cfg, state, rows):
    import torch
    from experiments.pilot_measurements import utility
    batches = [torch.tensor([r['tokens']], dtype=torch.long) for r in rows]
    return utility(split, cfg, state, batches, [])


def cell(args):
    import torch
    from huggingface_hub import snapshot_download
    from attacker import splitbench as sb
    from attacker.evidence import EvidenceLogger
    torch.set_num_threads(args.threads)
    ppath = Path(args.protocol).resolve()
    protocol = json.loads(ppath.read_text())
    if protocol['status'] != 'frozen':
        raise ValueError('Protocol is not frozen')
    for rel, want in protocol['implementation_sha256'].items():
        if digest(ROOT / rel) != want:
            raise ValueError('Implementation hash mismatch: ' + rel)
    data = read_inputs(protocol, ROOT / protocol['input'])
    root = Path(args.workdir).resolve()
    if root.exists():
        raise ValueError('Cell directory must be fresh; failed cells are never retried in place')
    root.mkdir(parents=True)
    config = {'span': args.span, 'start': args.start, 'defense': args.defense,
              'seed': args.seed, 'kind': args.kind, 'device': args.device,
              'protocol_sha256': digest(ppath), 'input_sha256': protocol['input_sha256'],
              'model_revision': protocol['model_revision']}
    if args.seed not in protocol['victim_seeds'] or args.span not in protocol['spans']:
        raise ValueError('Cell outside frozen matrix')
    if args.start not in protocol['starts'] or args.defense not in protocol['defenses']:
        raise ValueError('Cell outside frozen matrix')
    save(root / 'config.json', config)
    model_path = snapshot_download(protocol['model'], revision=protocol['model_revision'], local_files_only=True)
    control = args.kind == 'all_local'
    cfg = sb.BenchConfig(model=model_path, split='1' if control else str(max(1, args.start)),
        mode='finetune', steps=protocol['train_steps'], seq_len=protocol['seq_len'], batch_size=1,
        lr=protocol['lr'], seed=args.seed, device=args.device, dtype='fp32',
        defense='none' if control else 'quant', quant_bits=8 if args.defense == 'D0' else 4,
        quant_grad=not control, grad_clip=1.0 if not control and args.defense == 'D1' else 0.0,
        grad_noise=0.1 if not control and args.defense == 'D1' else 0.0,
        trainable_range=None if control else f'{args.start},{args.start+args.span}',
        capture_dir=str(root/'training-captures'), output=str(root/'training.json'), save_canonical=True)
    cfg.validate()
    torch.manual_seed(args.seed)
    model, _ = sb._load_model_and_tokenizer(cfg)
    split = sb.build_split(model, sb.SplitPlan.parse(cfg.split, n_layers=sb._n_layers_of(model)))
    permutation = torch.randperm(len(data['train']), generator=torch.Generator().manual_seed(args.seed)).tolist()
    batches = [torch.tensor([data['train'][i]['tokens']], dtype=torch.long) for i in permutation]
    measured = {}
    def measure(stage, split, cfg, state):
        measured[stage] = utility_rows(split, cfg, state, data['evaluation'])
        if stage == 'after':
            measured['documents'] = heldout_observations(split, cfg, state, data['evaluation'], root/'heldout-captures')
            # Enough to restore the changed victim parameters from the pinned base.
            changed = {name: parameter.detach().cpu() for name, parameter in model.named_parameters()
                       if parameter.requires_grad}
            torch.save(changed, root/'trained-parameters.pt')
    with EvidenceLogger(root/'training.json', node_role='p6-corrected/training') as evidence:
        evidence.event('prospective_protocol', config)
        with sb.CaptureWriter(cfg.capture_dir, 'training', root.name) as writer:
            sb.execute(cfg, split, writer, batches, evidence=evidence, measurement=measure)
    with EvidenceLogger(root/'measurements.json', node_role='p6-corrected/heldout') as evidence:
        evidence.event('heldout_observation', {'documents': len(measured['documents']),
            'optimizer_updates': 0, 'partition': 'evaluation', 'config': config})
        save(root/'measurements.json', measured)
    # Release victim memory before loading the public attack model.
    del split, model, batches
    import gc
    gc.collect()
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    torch.save(torch.tensor([r['tokens'] for r in data['calibration']], dtype=torch.long), root/'public-calibration.pt')
    if True:  # Undefended all_local attacks are the linked positive controls.
        commands = [[sys.executable, '-m', 'attacker', '--mode', 'training', '--attack', 'seam-inversion',
                    '--arm', 'vmi', '--model', model_path, '--split', cfg.split,
                    '--capture-dir', str(root/'heldout-captures'), '--canonical-pt', str(root/'heldout-captures/canonical.pt'),
                    '--samples', '256', '--seeds', str(protocol['attack_seed']),
                    '--calibration-pt', str(root/'public-calibration.pt'), '--calibration-seqs', '32', '--device', args.device, '--output', str(root/'vmi.json'), '--evidence']]
        for token in protocol['property_tokens']:
            commands.append([sys.executable, '-m', 'attacker', '--mode', 'training', '--attack', 'label-scoring',
                '--capture-dir', str(root/'heldout-captures'), '--canonical-pt', str(root/'heldout-captures/canonical.pt'),
                '--property', f'contains-token:{token}', '--output', str(root/f'label-{token}.json'), '--evidence'])
        for cmd in commands:
            with (root/(Path(cmd[cmd.index('--output')+1]).stem+'.log')).open('w') as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=protocol['attack_timeout_s'])
        attack = json.loads((root/'vmi.json').read_text())
        results = attack['results']
        if (len(results) != 256 or {r['step'] for r in results} != set(range(256))
                or any('error' in r or r.get('n_tokens') != 32 or 'per_sequence' not in r for r in results)):
            raise ValueError('VMI missing heldout documents')
        for token in protocol['property_tokens']:
            out = json.loads((root/f'label-{token}.json').read_text())
            scores = [r for r in out['results'] if r.get('kind') == 'property_score']
            if len(scores) != 1 or scores[0]['n_present'] + scores[0]['n_absent'] != 256:
                raise ValueError('Property attack missing heldout documents')
    files = {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
    report = {**config, 'status': 'complete', 'files': files,
              'final_utility': measured['after'], 'frozen_utility': measured['before'] if control else None,
              'interpretation': 'Prospective corrected Mode-2 follow-up, not Mode-1/P5 evidence'}
    with EvidenceLogger(root/'cell.json', node_role='p6-corrected/cell') as evidence:
        evidence.event('cell_complete', {**config, 'files': len(files)})
        save(root/'cell.json', report)
    verify_cell(root, digest(ppath))
    print('CELL_VERIFIED', root, flush=True)


def grid(args):
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    root = Path(args.workdir).resolve()
    if root.exists() and not args.resume:
        raise ValueError('Grid directory exists; use --resume only for verified completed cells')
    root.mkdir(parents=True, exist_ok=True)
    journal = root/'run-index.jsonl'
    previous = read_journal(journal)
    if any(r['rc'] != 0 for r in previous):
        raise ValueError('Failed attempt preserved: use a fresh grid directory and explicit attempt reconciliation')
    matrix = [c for span in protocol['spans'] for c in order(protocol, span)] if args.span == 0 else order(protocol, args.span)
    if args.span == 0:
        random.Random(protocol['order_seed']).shuffle(matrix)
    items = [('trained', *c) for c in matrix]
    if args.controls:
        items = [('all_local', 2, 0, 'D0', seed) for seed in protocol['victim_seeds']] + items
    if args.canary:
        items = [('trained', args.span or 6, 1, 'D1', protocol['victim_seeds'][0])]
    completed = {r['tag'] for r in previous}
    manifest = {'protocol_sha256': digest(protocol_path), 'span': args.span,
        'node': args.node, 'canary': args.canary, 'order': [list(item) for item in items],
        'controls': args.controls, 'device': args.device, 'threads': args.threads}
    manifest_path = root/'grid-manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Resume configuration differs from initial grid manifest')
    else:
        save(manifest_path, manifest)

    for kind, span, start, defense, seed in items:
        name = f'{kind}-'+tag(span, start, defense, seed)
        target = root/name
        if name in completed:
            verify_cell(target, digest(protocol_path))
            continue
        if target.exists():
            raise ValueError('Unjournaled/incomplete cell directory; fresh attempt required: ' + str(target))
        command = [sys.executable, '-m', 'experiments.run_p6_corrected', 'cell', '--span', str(span),
            '--start', str(start), '--defense', defense, '--seed', str(seed), '--kind', kind,
            '--workdir', str(target), '--protocol', str(protocol_path), '--device', args.device,
            '--threads', str(args.threads)]
        started = time.monotonic()
        with (root/(name+'.log')).open('w') as output:
            try:
                rc = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                    timeout=protocol['cell_timeout_s']).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        if rc == 0:
            try:
                verify_cell(target, digest(protocol_path))
            except Exception as error:
                rc = 1
                print(str(error), flush=True)
        record = {'tag': name, 'kind': kind, 'rc': rc, 'wall_s': time.monotonic()-started,
                  'cell': str(target), 'protocol_sha256': digest(protocol_path)}
        append_journal(journal, record)
        print(json.dumps(record), flush=True)
        if rc:
            from attacker.evidence import EvidenceLogger
            failure = root/(name+'.failure.json')
            with EvidenceLogger(failure, node_role='p6-corrected/failure') as evidence:
                evidence.event('cell_failure', record)
                save(failure, {**record, 'log_sha256': digest(root/(name+'.log')), 'command': command})
            raise RuntimeError('Cell failed; comparative analysis blocked')
    save(root/'grid-complete.json', {'protocol_sha256': digest(protocol_path), 'count': len(items),
                                   'canary': args.canary, 'status': 'complete'})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['cell', 'grid'])
    ap.add_argument('--protocol', default=str(DEFAULT_PROTOCOL))
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--span', type=int, choices=[0,2,4,6], required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--defense', choices=['D0','D1'], default='D0')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--kind', choices=['trained','all_local'], default='trained')
    ap.add_argument('--node', default='local')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--controls', action='store_true')
    ap.add_argument('--canary', action='store_true')
    args = ap.parse_args()
    (cell if args.action == 'cell' else grid)(args)


if __name__ == '__main__':
    main()
