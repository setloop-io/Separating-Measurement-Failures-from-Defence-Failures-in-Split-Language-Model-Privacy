"""Compare independent SIPIT kernels with a pinned external author checkout.

The author code is imported from --reference; none is vendored in this project.
Noiseless fixtures and optional public calibration prefixes are used; no victim samples.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacker.evidence import EvidenceLogger, file_sha256

REFERENCE_COMMIT = '820683156b7257313046a4fb3c492e52519525b7'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot-root', type=Path, help='also check two-token public pretrained calibration prefixes')
    args = parser.parse_args()
    actual = subprocess.check_output(['git', '-C', str(args.reference), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(args.reference), 'status', '--porcelain', '--', 'src'], text=True)
    if actual != REFERENCE_COMMIT or dirty:
        raise ValueError('reference source must match the pinned author commit')
    if args.output.exists():
        raise ValueError('use a fresh fidelity artifact path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.reference.resolve()))
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from src.algorithm.SIPIT import SIPIT
    from attacker.attacks.seam_inversion import _head_embed_forward, _load_head_pt
    from attacker.reconstruction_baselines import MeteredHead, sipit_reconstruct
    from attacker.splitbench import SplitPlan, build_split

    class Adapter:
        def __init__(self, split):
            self.split, self.device = split, split.device

        def get_input_embeddings(self):
            return self.split.wte

        def __call__(self, input_ids=None, inputs_embeds=None, **kwargs):
            h = (self.split.head_forward(input_ids) if input_ids is not None
                 else _head_embed_forward(self.split, inputs_embeds))
            return SimpleNamespace(hidden_states=(h,))

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return ':'.join(map(str, ids))

    def check_case(split, ids, seed, lr, budget, tier):
        tokens = torch.tensor([ids])
        depth = split.ranges[0][1]
        with torch.no_grad():
            h = split.head_forward(tokens)
            full = split.model(tokens, output_hidden_states=True, use_cache=False).hidden_states[depth]
        parity = float((h - full).abs().max())
        if not torch.allclose(h, full, atol=1e-6, rtol=1e-5):
            raise AssertionError('head adapter does not match HF hidden states')
        ours, diagnostic = sipit_reconstruct(MeteredHead(split), h, split.wte.weight.detach(),
                                             steps=budget, seed=seed, step_size=lr)
        # Exact bounded recovery first prevents an unbounded reference search
        # on an accidentally mismatched pretrained fixture.
        if ours[0].tolist() != ids:
            raise AssertionError('bounded fidelity fixture did not recover exactly')
        author = SIPIT(log_dir=str(args.output.parent / 'author-log'), log_name=None)
        torch.manual_seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            _, author_ids, author_iterations, _ = author.find_prompt(model=Adapter(split),
                tokenizer=Tokenizer(), layer_idx=0, target_hidden_states=h[0], step_size=lr)
        counts = [row[0] for row in diagnostic['iterations_by_position']]
        case = {'tier': tier, 'vocab': len(split.wte.weight), 'lr': lr, 'depth': depth, 'seed': seed,
                'author_prediction': author_ids, 'our_prediction': ours[0].tolist(),
                'author_iterations': author_iterations, 'our_iterations': counts,
                'head_parity_max_error': parity}
        if author_ids != ours[0].tolist() or author_iterations != counts:
            raise AssertionError(case)
        return case

    torch.set_num_threads(1)
    result = {'reference_commit': actual, 'cases': [], 'status': 'RUNNING',
              'verification_script_sha256': file_sha256(__file__),
              'implementation_sha256': file_sha256(Path(__file__).resolve().parents[1] / 'attacker/reconstruction_baselines.py'),
              'reference_files': {str(p.relative_to(args.reference)): file_sha256(p)
                                  for p in sorted((args.reference / 'src').rglob('*.py'))}}
    with EvidenceLogger(args.output) as evidence:
        try:
            for vocab, lr in [(64, 1.0), (256, .0001)]:
                torch.manual_seed(20260908)
                cfg = GPT2Config(vocab_size=vocab, n_layer=7, n_embd=24, n_head=4, n_positions=32,
                                 resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
                cfg._attn_implementation = 'eager'
                model = GPT2LMHeadModel(cfg).eval().requires_grad_(False)
                for depth in [3, 6]:
                    split = build_split(model, SplitPlan.parse(str(depth), n_layers=7))
                    for seed, ids in [(11, [3, 17, 45, 2]), (22, [58, 0, 12, 31])]:
                        case = check_case(split, ids, seed, lr, vocab, 'tiny_noiseless')
                        result['cases'].append(case)
                        evidence.event('parity_pass', case)
            if args.pilot_root:
                for depth in [3, 6]:
                    folder = args.pilot_root / f'split{depth}'
                    head_path, cal_path = folder / 'public-head-11.pt', folder / 'calibration.pt'
                    split, _ = _load_head_pt(str(head_path), 'cpu')
                    split.model.requires_grad_(False)
                    calibration = torch.load(cal_path, map_location='cpu', weights_only=True)
                    case = check_case(split, calibration[0, :2].tolist(), 11, 1.0, 128, 'pretrained_public_calibration')
                    case.update(head_sha256=file_sha256(head_path), calibration_sha256=file_sha256(cal_path))
                    result['cases'].append(case)
                    evidence.event('parity_pass', case)
            result['status'] = 'COMPLETE'
        except BaseException as error:
            result.update(status='FAILED', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'status': result['status'], 'cases': len(result['cases'])}))


if __name__ == '__main__':
    main()
