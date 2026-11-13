"""Fresh co-trained defense run and final-state inference white-box addendum.

Never reclassifies historical training captures. Torch imports are runtime only.
The attack is PR35's corrected token-embedding -> victim head -> encoder
objective, adapted explicitly to final-head exports and inference/prefill wires.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

SCHEMA = 'llm_attacker.whitebox35_cotrain_final_inference.v1'


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def tensor_state_digest(model, defense):
    """Digest parameters and persistent buffers without pickle-format dependence."""
    import torch
    h = hashlib.sha256()
    parts = {'victim/' + k: v for k, v in model.state_dict().items()}
    parts.update({'encoder/' + k: v for k, v in defense.state_dict.items()})
    parts.update({'geometry/' + k: v for k, v in (defense.geometry or {}).items()
                  if torch.is_tensor(v)})
    for name, tensor in sorted(parts.items()):
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode()); h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    h.update(json.dumps(defense.config, sort_keys=True).encode())
    h.update(json.dumps({k:v for k,v in (defense.geometry or {}).items()
                         if not torch.is_tensor(v)}, sort_keys=True).encode())
    return h.hexdigest()


def observe_final_state(split, defense, batches, output):
    """Observe immutable final modules. Stored fp16 wire equals checkpoint replay."""
    import torch
    from attacker import splitbench as sb
    from attacker import captures
    from attacker.attacks.seam_inversion import save_head_pt, _load_head_pt
    from attacker.trained_defenses import TrainedBottleneck
    from attacker.evidence import file_sha256
    output = Path(output)
    capdir = output / 'inference-captures'
    if capdir.exists():
        raise ValueError('inference capture directory must be fresh')
    modes = [(m, m.training) for m in split.model.modules()]
    before = tensor_state_digest(split.model, defense)
    grad_before = {k: None if p.grad is None else p.grad.detach().clone()
                   for k, p in split.model.named_parameters()}
    head_path = output / 'final-victim-head.pt'
    boundary = split.plan.boundaries[0] if hasattr(split.plan, 'boundaries') else None
    if boundary is None:
        raise ValueError('Explicit GPT2 split boundary unavailable')
    save_head_pt(str(head_path), split.model, boundary)
    defense.save(output / 'final-defense')
    # Full victim state permits independent final-utility reproduction too.
    torch.save({'config': split.model.config.to_dict(),
                'state_dict': {k: v.detach().cpu() for k, v in split.model.state_dict().items()}},
               output / 'final-victim.pt')
    replay_split, _ = _load_head_pt(str(head_path), str(split.device))
    replay_defense = TrainedBottleneck.load(output / 'final-defense')
    canonical, checks = [], []
    try:
        split.model.eval()
        with torch.no_grad(), sb.CaptureWriter(capdir, 'inference', 'whitebox35-final-inference') as writer:
            for i, batch in enumerate(batches):
                ids = batch.to(split.device)
                hidden = split.head_forward(ids)
                wire = defense.apply(hidden)
                _, path = writer.write_inference(wire, request_seq=i, phase='prefill', position=0, epoch=0)
                replay = replay_defense.apply(replay_split.head_forward(ids)).detach().cpu().half()
                stored = captures.load_tensor(path)
                if not torch.equal(stored, replay):
                    raise ValueError('Saved final head/encoder do not reproduce stored wire')
                canonical.append(ids.cpu())
                checks.append({'request_seq': i, 'checkpoint_replay_exact_fp16': True,
                               'max_abs_error': float((stored.float()-replay.float()).abs().max())})
        torch.save(torch.cat(canonical), capdir / 'canonical.pt')
    finally:
        for m, mode in modes: m.training = mode
    after = tensor_state_digest(split.model, defense)
    if before != after: raise ValueError('Observation mutated victim or encoder state')
    for k, p in split.model.named_parameters():
        old = grad_before[k]
        if (old is None) != (p.grad is None) or (old is not None and not torch.equal(old, p.grad)):
            raise ValueError('Observation changed parameter gradients')
    return {'observation_phase': 'post-training inference', 'capture_mode': 'inference',
            'capture_phase': 'prefill', 'head_path': 'final-victim-head.pt',
            'checkpoint': 'final-defense/trained_bottleneck_encoder.pt',
            'captures_dir': 'inference-captures', 'state_before_sha256': before,
            'state_after_sha256': after, 'optimizer_updates_during_observation': 0,
            'parameter_gradients_unchanged': True, 'replay_checks': checks,
            'final_head_sha256': file_sha256(head_path),
            'checkpoint_sha256': file_sha256(output/'final-defense/trained_bottleneck_encoder.pt'),
            'capture_sha256': {p.relative_to(output).as_posix(): file_sha256(p)
                               for p in sorted(capdir.iterdir()) if p.is_file()}}


def adaptive_final_inference(args, root, entry, budgets):
    """PR35 objective with explicit final-head/inference entry, no phase relabel."""
    import torch
    from attacker import captures
    from attacker.attacks import seam_inversion as si
    from attacker.reconstruction_baselines import MeteredHead
    from attacker.trained_defenses import TrainedBottleneck
    root = Path(root)
    from attacker.evidence import file_sha256
    if entry['capture_mode'] != 'inference' or entry['capture_phase'] != 'prefill':
        raise ValueError('This adapter only evaluates explicit final-state inference captures')
    for rel, expected in {entry['head_path']:entry['final_head_sha256'],
                          entry['checkpoint']:entry['checkpoint_sha256'],
                          **entry['capture_sha256']}.items():
        if file_sha256(root/rel) != expected: raise ValueError('Entry artifact hash mismatch: '+rel)
    defense = TrainedBottleneck.load(root / entry['checkpoint'])
    split, embedding = si._load_head_pt(str(root / entry['head_path']), args.device)
    split.model.requires_grad_(False)
    embedding = embedding.detach(); head = MeteredHead(split)
    records = [(m,p) for m,p in captures.scan_captures(str(root / entry['captures_dir']), 'inference')
               if m.get('phase') == 'prefill'][:budgets['attack_sequences']]
    if not records: raise ValueError('No inference/prefill captures')
    wire = torch.cat([captures.load_tensor(p) for _,p in records]).to(args.device)
    truth = torch.load(root / entry['captures_dir'] / 'canonical.pt', map_location=args.device,
                       weights_only=True)[:len(records)]
    z = wire[..., :entry['dim']].detach(); n,length,_ = wire.shape
    with torch.random.fork_rng(devices=[]), head.measure() as work:
        torch.manual_seed(entry['seed'] ^ 0xE2E)
        init = torch.randint(len(embedding), (n,length))
        variable = embedding[init.to(args.device)].detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([variable], lr=args.optimize_lr)
        losses = []
        for _ in range(budgets['sipit_steps']):
            opt.zero_grad()
            loss = (defense.encode(head(embeddings=variable))-z).square().mean()
            if not torch.isfinite(loss): raise ValueError('Non-finite adaptive objective')
            loss.backward(); head.backward(n,length)
            if variable.grad is None or not torch.isfinite(variable.grad).all():
                raise ValueError('Missing/non-finite true input gradients')
            opt.step(); losses.append(float(loss.detach()))
        with torch.no_grad():
            pred = (embedding.square().sum(-1)[None,None] + variable.square().sum(-1,keepdim=True)
                    -2*variable@embedding.T).argmin(-1)
            head.work.vocabulary_dot_products += n*length*len(embedding)
    if truth.shape != pred.shape: raise ValueError('Canonical/prediction alignment mismatch')
    predictions = {'predicted_ids': pred.cpu().tolist(), 'truth_ids': truth.cpu().tolist(),
                   'truth_use': 'scoring only, after optimisation'}
    return {'accuracy': float((pred==truth).float().mean()), 'n_tokens': truth.numel(),
            'n_sequences': n, 'objective': 'MSE(encoder(final_victim_head(token_embeddings)), released_z)',
            'steps_per_wire': budgets['sipit_steps'], 'loss_trace': losses,
            'work': vars(work).copy(), 'predictions': predictions,
            'observation_phase': 'post-training inference', 'final_victim_head_granted': True}


def run_cell(args, base, encoded, output, partition_manifest=None):
    """Fresh training; encoded partitions supplied by CLI or tiny tests."""
    import torch
    from attacker import splitbench as sb
    from attacker.evidence import EvidenceLogger, file_sha256
    from experiments.run_bottleneck_sweep import _bench_config
    from experiments.pilot_measurements import utility
    output = Path(output)
    if not 1 <= args.depth < base.config.n_layer: raise ValueError('Invalid GPT2 split depth')
    if not 0 < args.dim <= base.config.n_embd: raise ValueError('Invalid bottleneck dimension')
    if args.sipit_steps < 1 or args.attack_sequences < 1: raise ValueError('Attack budgets must be positive')
    if args.train_steps < args.tb_cotrain_interval: raise ValueError('Training must include a co-training update')
    if output.exists(): raise ValueError('Output must be fresh; historical runs are immutable')
    output.mkdir(parents=True)
    torch.save({k:torch.cat(v).cpu() for k,v in encoded.items()}, output/'encoded-inputs.pt')
    if partition_manifest is not None: save(output/'partitions.json', partition_manifest)
    torch.manual_seed(args.seed)
    model = copy.deepcopy(base).to(args.device)
    split = sb.build_split(model, sb.SplitPlan.parse(str(args.depth), n_layers=model.config.n_layer))
    cfg = _bench_config(args, args.depth, args.seed, 'tb-cotrain', args.dim, output)
    # Training captures stay separate and are never fed to the new attack.
    cfg.capture_dir = str(output / 'training-captures')
    retained = {}; measured = {}
    def measurement(stage, current, config, state):
        measured[stage] = utility(current, config, state, encoded['utility_eval'], encoded['calibration'])
        if stage == 'after': retained['state'] = state
    order = list(range(len(encoded['victim_train']))); random.Random(args.seed).shuffle(order)
    try:
        with EvidenceLogger(output / 'run.json', node_role='whitebox35/cotrain-final') as evidence:
            evidence.event('changed_observation_contract', {'phase':'post-training inference',
                           'access':'final victim head and final encoder granted; stronger than historical public-head access'})
            with sb.CaptureWriter(cfg.capture_dir, 'training', output.name) as writer:
                stats = sb.execute(cfg, split, writer, [encoded['victim_train'][i] for i in order],
                                   evidence=evidence, measurement=measurement)
        state = retained['state']; defense = state['tb']
        if state['tb_cotrain'].updates < 1: raise ValueError('No co-training update occurred')
        selected = encoded['utility_eval'][:args.attack_sequences]
        if len(selected) != args.attack_sequences: raise ValueError('Insufficient held-out observation sequences')
        observation = observe_final_state(split, defense, selected, output)
        entry = {**observation, 'method':'tb-cotrain', 'split':args.depth, 'seed':args.seed, 'dim':args.dim,
                 'path':'.', 'access_condition':'final-state white-box inference; changed phase and stronger final-head grant'}
        save(output / 'entry.json', entry)
        attack = adaptive_final_inference(args, output, entry,
                  {'attack_sequences': args.attack_sequences, 'sipit_steps': args.sipit_steps})
        predictions = attack.pop('predictions'); save(output / 'predictions.json', predictions)
        report = {'schema':SCHEMA, 'status':'COMPLETE', 'tier':'cotrain-final-inference',
                  'historical_training_cells_evaluated':False, 'entry':entry,
                  'utility_before':measured['before'], 'utility_after':measured['after'],
                  'training':stats, 'cotrain_updates':state['tb_cotrain'].updates,
                  'implementation_sha256': {str(p):file_sha256(p) for p in [
                      Path(__file__),Path(sb.__file__)]},
                  'attack':attack, 'configuration':vars(args),
                  'files':{p.relative_to(output).as_posix():file_sha256(p)
                           for p in sorted(output.rglob('*')) if p.is_file()}}
        with EvidenceLogger(output/'cell.json', node_role='whitebox35/cotrain-final-inference') as evidence:
            evidence.event('final_state_match', observation)
            save(output/'cell.json', report)
        verify_cell(output)
        return report
    except Exception as error:
        with EvidenceLogger(output/'failure.json', node_role='whitebox35/cotrain-failure') as evidence:
            evidence.event('failure', {'type':type(error).__name__, 'error':str(error)})
            save(output/'failure.json', {'schema':SCHEMA,'status':'FAILED','error':repr(error),
                                        'historical_training_cells_evaluated':False})
        raise


def verify_cell(root):
    """Hash/evidence-only packaging check; no torch, model loads or GPU work."""
    from attacker.evidence import file_sha256, verify_evidence
    root = Path(root).resolve()
    report = json.loads((root/'cell.json').read_text())
    if report.get('schema') != SCHEMA or report.get('status') != 'COMPLETE':
        raise ValueError('Co-trained inference cell is not complete')
    if report.get('historical_training_cells_evaluated') is not False:
        raise ValueError('Historical capture scope is not preserved')
    entry = report['entry']
    if entry['capture_mode'] != 'inference' or entry['capture_phase'] != 'prefill':
        raise ValueError('Unexpected observation phase')
    if entry['state_before_sha256'] != entry['state_after_sha256']:
        raise ValueError('Observation state changed')
    for rel, expected in report['files'].items():
        path = (root/rel).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_sha256(path) != expected:
            raise ValueError('Cell artifact hash mismatch: '+rel)
    for manifest in sorted(root.rglob('*.evidence.json')):
        ok, problems = verify_evidence(manifest)
        if not ok: raise ValueError('Evidence validation failed: '+str(problems))
    return report


def build_parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True);ap.add_argument('--depth',type=int,required=True)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--dim',type=int,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--quick',action='store_true')
    ap.add_argument('--model',default='openai-community/gpt2');ap.add_argument('--revision',default='607a30d783dfa663caf39e06633721c8d4cfcd7e')
    ap.add_argument('--corpus',default='results/pilot-gpt2-wikitext2/inputs/pilot-corpus.jsonl')
    ap.add_argument('--manifest',default='experiments/validation_corpus_manifest.json')
    ap.add_argument('--data-seed',type=int,default=2026);ap.add_argument('--seq-len',type=int,default=32)
    ap.add_argument('--train-steps',type=int,default=64);ap.add_argument('--lr',type=float,default=5e-5)
    ap.add_argument('--tb-hidden',type=int,default=256);ap.add_argument('--tb-probe-hidden',type=int,default=128)
    ap.add_argument('--tb-minimax-steps',type=int,default=200);ap.add_argument('--tb-cotrain-interval',type=int,default=8)
    ap.add_argument('--calibration-seqs',type=int,default=32)
    ap.add_argument('--attack-sequences',type=int,default=32);ap.add_argument('--sipit-steps',type=int,default=128)
    ap.add_argument('--optimize-lr',type=float,default=.05);ap.add_argument('--threads',type=int,default=1)
    return ap


def main():
    args=build_parser().parse_args()
    import torch
    from experiments.run_defended_pilot import _tiny_model,_synthetic_docs,_partition,assert_development_corpus_disjoint
    torch.set_num_threads(args.threads)
    if args.quick:
        docs=_synthetic_docs(n=max(18,3*args.attack_sequences),seq_len=args.seq_len);base=_tiny_model(args.seed);tok=None
    else:
        from transformers import AutoTokenizer,AutoModelForCausalLM
        assert_development_corpus_disjoint(args.corpus,args.manifest)
        docs=[json.loads(line) for line in Path(args.corpus).read_text().splitlines() if line.strip()]
        tok=AutoTokenizer.from_pretrained(args.model,revision=args.revision)
        base=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision)
    if base.config.model_type!='gpt2':raise ValueError('This adapter currently supports GPT2 only')
    parts=_partition(docs,args.data_seed);encoded={}
    for name,rows in parts.items():
        encoded[name]=[]
        for r in rows:
            ids=r['tokens'] if args.quick else tok.encode(r['text'],add_special_tokens=False)
            if len(ids)<args.seq_len:raise ValueError('Source document shorter than requested window')
            encoded[name].append(torch.tensor([ids[:args.seq_len]],dtype=torch.long))
    report=run_cell(args,base,encoded,args.output,
        {k:[{'id':r['id'],'text_sha256':hashlib.sha256(r['text'].encode()).hexdigest()} for r in rows] for k,rows in parts.items()})
    print(json.dumps({'status':report['status'],'cell':str(Path(args.output)/'cell.json')}))

if __name__=='__main__':main()
