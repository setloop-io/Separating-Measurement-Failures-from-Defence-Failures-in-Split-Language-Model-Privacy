#!/usr/bin/env python3
"""Generate SIP/b/f/bf predictions from a hash-pinned, label-free manifest.

This engineering runner does not issue fidelity approval or launch a campaign.
Prepare public_model.pt as {config, state_dict}; inverter as bisr_gru.v1;
observations use bisr.observations.v1. No truth file is accepted by this process.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacker.bisr_fidelity import (GRU_FORMAT, sha256, gru_class, granted_split,
                                    predict_arms, write_predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Prediction output already exists')
    manifest = json.loads(args.manifest.read_text())
    expected = {'schema', 'public_model', 'inverter', 'observations', 'knowledge',
                'head', 'boundary', 'bre'}
    if set(manifest) != expected or manifest['schema'] != 'bisr.prediction_manifest.v1':
        raise ValueError('Unknown manifest schema/fields (truth is forbidden)')
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def pinned(item):
        if set(item) != {'path', 'sha256'}:
            raise ValueError('Each input requires path and SHA-256 only')
        path = args.manifest.parent / item['path']
        if sha256(path) != item['sha256']:
            raise ValueError('Input digest mismatch: ' + str(path))
        return path

    public_path = pinned(manifest['public_model'])
    blob = torch.load(public_path, map_location='cpu', weights_only=True)
    if set(blob) != {'config', 'state_dict'}:
        raise ValueError('Public model bundle requires config and state_dict')
    config = GPT2Config(**blob['config'])
    config._attn_implementation = 'eager'
    model = GPT2LMHeadModel(config)
    model.load_state_dict(blob['state_dict'], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Non-finite public model')
    model.to(args.device).eval()
    head = manifest['head']
    split = granted_split(model, manifest['boundary'], knowledge=manifest['knowledge'],
        head_path=pinned(head) if head else None,
        head_sha256=head['sha256'] if head else None)
    checkpoint = torch.load(pinned(manifest['inverter']), map_location='cpu', weights_only=True)
    if checkpoint['format'] != GRU_FORMAT:
        raise ValueError('Prospective runner requires a GRU checkpoint')
    arch = checkpoint['arch']
    if arch['hidden_size'] != 256 or arch['dropout'] != 0.1:
        raise ValueError('Prospective GRU architecture mismatch')
    if arch['vocab'] != model.config.vocab_size or arch['seam_dim'] != model.config.n_embd:
        raise ValueError('Inverter/model shape mismatch')
    inverter = gru_class()(**arch).to(args.device)
    inverter.load_state_dict(checkpoint['state_dict'], strict=True)
    observations = torch.load(pinned(manifest['observations']), map_location='cpu', weights_only=True)
    bre = manifest['bre']
    if set(bre) != {'gm_iters', 'gm_lr', 'gm_beta', 'gm_tau', 'sm_iters', 'sm_lr', 'sm_wd', 'opt_batch'}:
        raise ValueError('All BRE budgets must be explicit')
    record = predict_arms(split, inverter, observations, SimpleNamespace(**bre))
    record['input_manifest_sha256'] = sha256(args.manifest)
    record['runner_sha256'] = sha256(__file__)
    record['implementation_sha256'] = {
        name: sha256(Path(__file__).resolve().parents[1] / name) for name in
        ('attacker/bisr_fidelity.py', 'attacker/attacks/bisr_baseline.py')}
    record['budgets'] = bre
    digest = write_predictions(args.output, record)
    print(json.dumps({'predictions': str(args.output), 'sha256': digest,
                      'launch_admitted': False}))


if __name__ == '__main__':
    main()
