"""Paired controlled validation. Draft permits synthetic smoke only until frozen."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacker.evidence import EvidenceLogger, file_sha256

HELDOUT_ACCESS_JOURNAL = Path('results/heldout_access.journal')


def record_heldout_access(output, journal=HELDOUT_ACCESS_JOURNAL):
    """Journal the first read of held-out evaluation outcomes; re-freeze is refused afterwards."""
    from datetime import datetime, timezone
    journal = Path(journal)
    journal.parent.mkdir(parents=True, exist_ok=True)
    entry = {'accessed_at': datetime.now(timezone.utc).isoformat(), 'output': str(output)}
    with journal.open('a') as handle:
        handle.write(json.dumps(entry) + '\n')


def capture_defenses(protocol):
    """Defended-capture defenders scheduled by the protocol (empty for v1)."""
    spec = protocol.get('capture_defenses') or {}
    return list(spec.get('defended', []))


#: Defenses this runner can execute defended-capture arms for, mapped to the
#: provenance sidecar filename the splitbench defense writes into the run's
#: capture dir (attacker/trained_defenses.py).
DEFENSE_PROVENANCE = {'replica-median': 'replica_median_provenance.json',
                      'secret-mixup': 'secret_mixup_provenance.json'}


def defended_arms(protocol, defense):
    """Training-mode defended arms scheduled for one defender."""
    return [f'forward_k{k}@{defense}' for k in protocol['candidate_counts']] + \
           [f'sipit_steps{s}@{defense}' for s in protocol['sipit_steps']]


def arms(protocol, mode):
    out = [f'forward_k{k}' for k in protocol['candidate_counts']]
    out += [f'sipit_steps{s}' for s in protocol['sipit_steps']]
    if mode == 'training':
        out += ['backward']
        out += [f'joint_{c}_k{k}' for k in protocol['candidate_counts']
                for c in protocol['gradient_conditions'] + ['random']]
        for defense in capture_defenses(protocol):
            out += [f'forward_k{k}@{defense}' for k in protocol['candidate_counts']]
            out += [f'sipit_steps{s}@{defense}' for s in protocol['sipit_steps']]
    return out


def expected_cell_ids(p):
    """Full logical cell-id matrix of the protocol, in canonical execution order."""
    ids=[]
    for seed in p['victim_seeds']:
        for depth in p['depths']:
            for knowledge in p['knowledge']:
                for attack_seed in p['attack_seeds']:
                    for mode in ('training','inference'):
                        for arm in arms(p,mode):
                            ids.append(f'{seed}-{depth}-{knowledge}-{attack_seed}-{mode}-{arm}')
    return ids


def shard_assignment(p, count):
    """Deterministic, communication-free partition of the cell matrix.

    Assignment rule: the physical execution roots (cell ids with reused
    inference copies folded into their training-mode origin) are sorted;
    root at sorted position i belongs to shard i % count. Reused inference
    cells (execution.reuse_identical_activation_modes) are co-assigned with
    their physical origin so a shard never references a measurement it did
    not run. Cell ids sort across depths, so interleaving sorted indices
    spreads every depth, seed, knowledge condition and arm over every shard
    — depth is never confounded with host. Any host can reproduce the
    partition from the protocol alone.
    """
    if type(count) is not int or count<1: raise ValueError('shard count must be a positive integer')
    ids=expected_cell_ids(p)
    roots=sorted({cid.replace('-inference-','-training-') for cid in ids})
    root_shard={root:i%count for i,root in enumerate(roots)}
    return {cid:root_shard[cid.replace('-inference-','-training-')] for cid in ids}


def validate_protocol(p,smoke=False):
    if p.get('schema')!='llm_attacker.validation.v1' or p.get('status') not in ('draft_for_review','frozen'):
        raise ValueError('invalid protocol schema/status')
    if type(p.get('version',1)) is not int or p.get('version',1)<1: raise ValueError('invalid protocol version')
    for key in ('depths','victim_seeds','attack_seeds','candidate_counts','sipit_steps','gradient_conditions','knowledge'):
        if not p[key] or len(p[key])!=len(set(p[key])): raise ValueError('empty/duplicate protocol axis')
    if p['knowledge']!=['public','exact'] or p['gradient_conditions']!=['true','shuffled','noise_0.1','noise_1.0','bits_8','bits_4']:
        raise ValueError('required knowledge/gradient controls missing or changed')
    for key in ('attack_batch_size','sequence_length','train_steps','beam_width','minimum_documents'):
        if type(p[key]) is not int or p[key]<1: raise ValueError('positive integer budget required')
    if not smoke and p['status']=='frozen':
        pin=p.get('frozen_inputs',{})
        for field in ('tokenized_sha256','name_annotations_sha256','curation_manifest_sha256'):
            value=pin.get(field,'')
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('frozen protocol requires complete input hash bindings')
        if pin.get('partition_counts')!={'calibration':32,'train':128,'evaluation':256}:
            raise ValueError('frozen partition counts must be32/128/256')
    if p['sequence_length']<2: raise ValueError('next-token loss requires at least two tokens')
    if any(type(k) is not int or k<2 or k%2 for k in p['candidate_counts']): raise ValueError('even candidate budgets required')
    if len(p['victim_seeds'])<(2 if smoke else 10) or p['minimum_documents']<(2 if smoke else 256):
        raise ValueError('insufficient prespecified sampling units')
    primary=p['primary']
    if primary['metric']!='token_accuracy' or primary['mode']!='training' or primary['depth'] not in p['depths'] or primary['knowledge'] not in p['knowledge']:
        raise ValueError('unsupported primary comparison')
    if primary['left'] not in arms(p,'training') or primary['right'] not in arms(p,'training'):
        raise ValueError('primary arms missing')
    if not 0<=primary['practical_margin']<=1: raise ValueError('invalid practical margin')
    spec = p.get('capture_defenses')
    if spec is not None:
        defended = spec.get('defended')
        if not isinstance(defended, list) or not defended or len(defended) != len(set(defended)):
            raise ValueError('invalid capture_defenses axis')
        if any(type(x) is not str or not x or x == 'none' or '@' in x for x in defended):
            raise ValueError('invalid capture_defenses axis')
    if not 0<p['statistics']['alpha']<1 or type(p['statistics']['bootstrap_replicates']) is not int or p['statistics']['bootstrap_replicates']<2:
        raise ValueError('invalid bootstrap settings')


def validate_data(data, protocol, vocab, smoke=False):
    from attacker.reconstruction_metrics import reconstruction_metrics
    seen_ids, seen_tokens, source_ids, grams_seen = set(), set(), set(), []
    for partition in ('calibration', 'train', 'evaluation'):
        rows = data[partition]
        if not rows:
            raise ValueError('empty partition')
        for row in rows:
            tokens = row['tokens']
            if not isinstance(row['id'], str) or not row['id'].strip() or row['id'] in seen_ids:
                raise ValueError('duplicate/invalid document ID')
            if len(tokens) != protocol['sequence_length'] or any(type(x) is not int or not 0 <= x < vocab for x in tokens):
                raise ValueError('invalid token sequence')
            if tuple(tokens) in seen_tokens:
                raise ValueError('duplicate token prefix across corpus')
            if not smoke:
                source_id=row.get('source_document_id')
                if not isinstance(source_id,str) or not source_id or source_id in source_ids:
                    raise ValueError('distinct source_document_id required for each document')
                source_ids.add(source_id)
                grams={tuple(tokens[i:i+5]) for i in range(len(tokens)-4)}
                if grams and any(other and len(grams & other)/len(grams | other)>=0.8 for other in grams_seen):
                    raise ValueError('near-duplicate token prefixes; curate before outcomes')
                grams_seen.append(grams)
            seen_ids.add(row['id']); seen_tokens.add(tuple(tokens))
            reconstruction_metrics(tokens, tokens, row.get('annotations', []))
    if not smoke and protocol.get('status')=='frozen':
        if {key:len(data[key]) for key in ('calibration','train','evaluation')}!=protocol['frozen_inputs']['partition_counts']:
            raise ValueError('input partition counts differ from frozen sample')
    if not smoke and (len(data['calibration'])<protocol.get('minimum_calibration_documents',1) or len(data['train'])<protocol.get('minimum_training_documents',1)):
        raise ValueError('insufficient calibration or training documents')
    if len(data['evaluation']) < (2 if smoke else protocol['minimum_documents']):
        raise ValueError('insufficient evaluation documents')
    if not smoke and data.get('provenance', {}).get('untouched_evaluation') is not True:
        raise ValueError('held-out provenance declaration required')


def attack_observations(head, hidden, gradient, embedding, decoder, majority, arm, protocol, seed):
    """Truth-free attack boundary; bound peak memory with fixed public batch size."""
    import torch
    from attacker.reconstruction_baselines import forward_scores, backward_scores, candidate_ids, beam_reconstruct, sipit_reconstruct
    from attacker.validation_controls import transform_gradient, random_candidates
    arm = arm.split('@', 1)[0]  # defended arms share the undefended attack code path
    transformed=None; details={}; predictions=[]; batches=[]
    condition=None
    if arm.startswith('joint'):
        condition=arm[len('joint_'):].rsplit('_k',1)[0]
        if condition!='random': transformed,details=transform_gradient(gradient,condition,seed)
    for start in range(0,len(hidden),protocol['attack_batch_size']):
        stop=start+protocol['attack_batch_size']; h=hidden[start:stop]
        batch_details={}
        with torch.no_grad():
            fwd=forward_scores(h,embedding,decoder,head) if arm.startswith(('forward','joint')) else None
            if arm=='backward':
                pred=backward_scores(gradient[start:stop],embedding,head,majority).argmax(-1)
            elif arm.startswith(('forward','joint')):
                k=int(arm.rsplit('k',1)[1])
                if arm.startswith('forward'): candidates=candidate_ids(fwd,None,k,'forward')
                elif condition=='random': candidates=random_candidates(fwd,k,seed+start)
                else:
                    bwd=backward_scores(transformed[start:stop],embedding,head,majority)
                    candidates=candidate_ids(fwd,bwd,k,'joint')
                pred,batch_details=beam_reconstruct(head,h,candidates,protocol['beam_width'])
                batch_details['candidate_ids']=candidates.cpu().tolist()
        if arm.startswith('sipit'):
            pred,batch_details=sipit_reconstruct(head,h,embedding,steps=int(arm.split('steps')[1]),seed=seed+start)
        predictions.extend(pred.detach().cpu().tolist()); batches.append(batch_details)
    return predictions,dict(details,batches=batches)


def run_defended_capture(args, p, base, train_batches, evaluation, seed, depth, defense):
    """One defended victim run: splitbench defended training on the frozen
    victim-train partition, then post-defense wire capture of the evaluation
    documents under the post-training defense state.

    Reuses the splitbench defense wiring (execute/finetune ->
    make_defense_state/_apply_seam_defense); the defense transform is never
    reimplemented here. The training-order shuffle matches the undefended
    control's training order exactly (same Random(seed) draw over the same
    frozen partition). Returns the defended wire tensor plus the provenance,
    held-out utility and runtime records every defended cell must carry.
    Fails closed: a missing provenance sidecar raises."""
    import torch
    from attacker import splitbench as sb
    from attacker import trained_defenses as td
    out_dir = args.output / f'defended-{defense}-{seed}-{depth}'
    out_dir.mkdir(parents=True)
    victim = copy.deepcopy(base).to(args.device)
    split = sb.build_split(victim, sb.SplitPlan.parse(str(depth), n_layers=victim.config.n_layer))
    cfg = sb.BenchConfig(split=str(depth), mode='finetune', steps=p['train_steps'],
                         seq_len=p['sequence_length'], batch_size=1, lr=p['learning_rate'],
                         defense=defense, seed=seed, save_canonical=True, device=args.device,
                         capture_dir=str(out_dir / 'captures'),
                         output=str(out_dir / 'run.json')).validate()
    writer = sb.CaptureWriter(cfg.capture_dir, 'training', f'validation-v2-{defense}-{seed}-{depth}')
    if args.device == 'cuda': torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        retained_state = {}
        measured_utility = {}
        def retain(stage, current_split, current_cfg, state):
            from experiments.pilot_measurements import utility as matched_utility
            measured_utility[stage] = matched_utility(current_split, current_cfg, state,
                                                      [row[None] for row in evaluation], train_batches)
            if stage == 'after': retained_state.update(state)
        stats = sb.execute(cfg, split, writer, train_batches, measurement=retain,
                           capture_replica_input=True)
    finally:
        writer.close()
    if args.device == 'cuda': torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    prov_path = Path(cfg.capture_dir) / DEFENSE_PROVENANCE[defense]
    if not prov_path.is_file():
        raise RuntimeError(f'defense provenance sidecar missing: {prov_path}')
    provenance = {'path': str(prov_path.relative_to(args.output)), 'sha256': file_sha256(prov_path),
                  'schema': json.loads(prov_path.read_text()).get('schema')}
    victim.eval()
    # Retain the actual trained replica bodies and release state. For
    # secret-mixup the decoy pool is refilled from the training
    # blocks (the defense's documented recycle policy) before any evaluation
    # document is released.
    dstate = retained_state
    with torch.no_grad():
        if defense == 'secret-mixup':
            for ids in train_batches:
                dstate['mixup_ids'] = ids
                sb._apply_seam_defense(split.head_forward(ids.to(args.device)), cfg, dstate)
        utility_after = measured_utility['after']
        utility = {'utility_before': measured_utility['before'], 'utility_after': utility_after,
                   'contract': 'matched-final-state-v2; retained trained defender'}
        victim.requires_grad_(False)
        nll = utility_after['cross_entropy']
        wires = []
        for chunk in evaluation.split(p['attack_batch_size']):
            h = split.head_forward(chunk).detach()
            if defense == 'replica-median':
                # Integrity aggregation is downstream; this observer reads
                # the input seam that the participant actually receives.
                wire = h  # actual participant-visible input seam, not cloud output
            else:
                dstate['mixup_ids'] = chunk
                wire = sb._apply_seam_defense(h, cfg, dstate)
            wires.append(wire.detach())
    hidden = torch.cat(wires)
    obs_path = args.output / 'defended' / f'observations-{defense}-{seed}-{depth}.pt'
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'h': hidden.cpu()}, obs_path)
    return {'victim': victim, 'hidden': hidden,
            'training_seconds': training_seconds, 'training_stats': stats,
            'evaluation_nll': nll, 'utility': utility, 'provenance': provenance,
            'capture_dir': str(out_dir.relative_to(args.output)),
            'observations': {'path': str(obs_path.relative_to(args.output)),
                             'sha256': file_sha256(obs_path)}}


def reuse_activation_cell(cell):
    """Reference an identical measured attack; never label copied timing as new."""
    if cell['mode']!='training' or not cell['arm'].startswith(('forward_','sipit_')):
        raise ValueError('only activation-only training cells can be reused for inference')
    reused=copy.deepcopy(cell)
    reused.update(mode='inference',measurement_reused=True,reused_from=cell['cell_id'])
    reused['cell_id']=cell['cell_id'].replace('-training-','-inference-')
    return reused


def run(args):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from attacker.splitbench import build_split, SplitPlan
    from attacker.attacks import seam_inversion as si
    from attacker.reconstruction_baselines import MeteredHead
    from attacker.reconstruction_metrics import reconstruction_metrics
    p = json.loads(args.protocol.read_text())
    if not args.smoke and p['status'] != 'frozen':
        raise ValueError('freeze and review protocol before held-out execution')
    scheduled_defenses = capture_defenses(p)
    if scheduled_defenses and not args.smoke and p.get('measurement_contract') != 'matched-final-state-v2':
        raise ValueError('Defended measurement correction requires a reviewed protocol amendment '
                         'with measurement_contract=matched-final-state-v2 before held-out execution')
    unsupported = [d for d in scheduled_defenses if d not in DEFENSE_PROVENANCE]
    if unsupported:
        raise ValueError(f'defended-capture execution is implemented only for '
                         f'{sorted(DEFENSE_PROVENANCE)}; this protocol schedules '
                         f'{unsupported}; no execution may start')
    if args.output.exists():
        raise ValueError('output must be fresh')
    torch.set_num_threads(1)
    if args.smoke:
        p.update(depths=[1], victim_seeds=[101,202], attack_seeds=[17,29], sequence_length=4,
                 minimum_documents=3, train_steps=2, candidate_counts=[4], sipit_steps=[2])
        p['statistics']=dict(p['statistics'],bootstrap_replicates=1000)
        p['primary'] = dict(p['primary'], depth=1, left='joint_true_k4', right='forward_k4')
        rng = random.Random(882)
        data = {part: [{'id': f'{part}-{i}', 'tokens': [rng.randrange(64) for _ in range(4)],
                        'annotations': [{'kind':'number','start':1,'end':2}]} for i in range(3)]
                for part in ('calibration','train','evaluation')}
    else:
        if not args.input or not args.input_sha256 or file_sha256(args.input) != args.input_sha256:
            raise ValueError('externally pinned input SHA256 required')
        if args.input_sha256!=p.get('frozen_inputs',{}).get('tokenized_sha256'):
            raise ValueError('input SHA256 differs from frozen protocol')
        data = json.loads(args.input.read_text())
    validate_protocol(p,args.smoke)
    validate_data(data, p, 64 if args.smoke else 50257, args.smoke)
    shard_count=getattr(args,'shard_count',1); shard_index=getattr(args,'shard_index',0)
    if type(shard_count) is not int or shard_count<1 or type(shard_index) is not int or not 0<=shard_index<shard_count:
        raise ValueError('require shard-count >= 1 and 0 <= shard-index < shard-count')
    assignment=shard_assignment(p,shard_count)
    owned={cid for cid,shard in assignment.items() if shard==shard_index}
    args.output.mkdir(parents=True)
    if args.smoke:
        (args.output/'input.json').write_text(json.dumps(data))
    else:
        (args.output/'input.json').write_bytes(args.input.read_bytes())
    (args.output/'protocol-source.json').write_bytes(args.protocol.read_bytes())
    (args.output/'protocol-effective.json').write_text(json.dumps(p,sort_keys=True))
    if not args.smoke: record_heldout_access(args.output)
    artifact = args.output/'validation.json'
    report = {'schema':p['schema'], 'status':'RUNNING', 'evidence_tier':'engineering_smoke' if args.smoke else 'held_out_controlled_probe',
              'protocol':p, 'protocol_sha256':file_sha256(args.protocol), 'effective_protocol_sha256':file_sha256(args.output/'protocol-effective.json'), 'input_sha256':file_sha256(args.output/'input.json'),
              'supplied_input_sha256': args.input_sha256, 'cells':[], 'setup':[], 'victims':[], 'defended_victims':[],
              'shard':{'shard_count':shard_count,'shard_index':shard_index,'cell_ids':sorted(owned)},
              'runtime':{'torch':torch.__version__, 'cuda':torch.version.cuda, 'device':args.device}}
    def save():
        artifact.write_text(json.dumps(report, indent=2, allow_nan=False))
    def sync():
        if args.device == 'cuda': torch.cuda.synchronize()
    evaluation = torch.tensor([r['tokens'] for r in data['evaluation']], device=args.device)
    calibration = torch.tensor([r['tokens'] for r in data['calibration']], device=args.device)
    train = torch.tensor([r['tokens'] for r in data['train']], device=args.device)
    defended_failures = []
    with EvidenceLogger(artifact) as ev:
        save()
        try:
            for seed in p['victim_seeds']:
                torch.manual_seed(seed)
                started=time.perf_counter()
                base = (GPT2LMHeadModel(GPT2Config(vocab_size=64,n_embd=16,n_layer=2,n_head=2,n_positions=32,
                         resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.)) if args.smoke else
                        GPT2LMHeadModel.from_pretrained(p['model'],revision=p['revision']))
                acquisition_seconds=time.perf_counter()-started
                for depth in p['depths']:
                    si.save_head_pt(str(args.output/f'public-{seed}-{depth}.pt'),base,depth)
                victim = copy.deepcopy(base).to(args.device).train()
                optim = torch.optim.AdamW(victim.parameters(),lr=p['learning_rate'])
                order = list(range(len(train))); random.Random(seed).shuffle(order)
                sync(); started=time.perf_counter()
                for step in range(p['train_steps']):
                    ids=train[order[step % len(order)]][None]
                    optim.zero_grad(set_to_none=True)
                    loss=victim(ids,labels=ids).loss; loss.backward(); optim.step()
                sync(); training_seconds=time.perf_counter()-started
                victim.eval(); victim.requires_grad_(False)
                with torch.no_grad():
                    utility_loss=sum(float(victim(chunk,labels=chunk).loss)*len(chunk) for chunk in evaluation.split(p['attack_batch_size']))/len(evaluation)
                report['victims'].append({'seed':seed,'base_model_acquisition_seconds':acquisition_seconds,'training_seconds':training_seconds,'evaluation_nll':utility_loss})
                for depth in p['depths']:
                    vs=build_split(victim,SplitPlan.parse(str(depth),n_layers=victim.config.n_layer))
                    # Same post-training state, inputs and activation for both modes.
                    hs,gs=[],[]
                    for chunk in evaluation.split(p['attack_batch_size']):
                        with torch.no_grad(): h=vs.head_forward(chunk).detach()
                        leaf=h.clone().requires_grad_(True)
                        logits=vs.rest_forward(leaf)
                        loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),chunk[:,1:].reshape(-1),reduction='sum')/(len(evaluation)*(evaluation.shape[1]-1))
                        gs.append(torch.autograd.grad(loss,leaf)[0].detach()); hs.append(h)
                        del logits,leaf,loss
                    hidden=torch.cat(hs); gradient=torch.cat(gs)
                    torch.save({'h':hidden.cpu(),'g':gradient.cpu()},args.output/f'observations-{seed}-{depth}.pt')
                    sync(); started=time.perf_counter()
                    exact_path=args.output/f'exact-{seed}-{depth}.pt'
                    si.save_head_pt(str(exact_path),victim,depth)
                    export_seconds=time.perf_counter()-started
                    for knowledge in p['knowledge']:
                        path=args.output/f'{knowledge}-{seed}-{depth}.pt'
                        sync(); started=time.perf_counter()
                        split,embedding=si._load_head_pt(str(path),args.device)
                        split.model.requires_grad_(False); embedding=embedding.detach()
                        head=MeteredHead(split)
                        with torch.no_grad(): head(ids=calibration[:1])
                        sync(); loading=time.perf_counter()-started
                        with head.measure() as fit, torch.no_grad():
                            hc=head(ids=calibration).reshape(-1,embedding.shape[-1])
                            decoder,_=si._ridge_fit(hc,embedding[calibration.flatten()],0)
                        setup_id=f'{seed}-{depth}-{knowledge}'
                        setup={'id':setup_id,'loading_seconds':loading,'calibration_work':asdict(fit),
                               'exact_export_seconds':export_seconds if knowledge=='exact' else 0.,'head_sha256':file_sha256(path)}
                        report['setup'].append(setup)
                        majority=si.majority_control_token(calibration.tolist())
                        measured_activation_cells={}
                        for attack_seed in p['attack_seeds']:
                            for mode in ('training','inference'):
                                methods=[a for a in arms(p,mode) if '@' not in a]  # defended arms run against defended captures below
                                random.Random(seed+attack_seed+depth).shuffle(methods)
                                for arm in methods:
                                    name=f'{setup_id}-{attack_seed}-{mode}-{arm}'
                                    if name not in owned: continue  # cell belongs to another shard
                                    if mode=='inference' and p.get('execution',{}).get('reuse_identical_activation_modes'):
                                        cell=reuse_activation_cell(measured_activation_cells[attack_seed,arm])
                                        report['cells'].append(cell)
                                        ev.event('cell_reused',{k:v for k,v in cell.items() if k!='metrics'})
                                        continue
                                    with head.measure() as work:
                                        prediction,diagnostics=attack_observations(head,hidden,gradient if mode=='training' else None,
                                            embedding,decoder,majority,arm,p,attack_seed)
                                    prediction_path=args.output/f'{name}.json'
                                    prediction_path.write_text(json.dumps({'predictions':prediction,'document_ids':[r['id'] for r in data['evaluation']], 'diagnostics':diagnostics},allow_nan=False))
                                    metrics=[{'document_id':row['id'], **reconstruction_metrics(row['tokens'],predicted,row.get('annotations',[]))}
                                             for row,predicted in zip(data['evaluation'],prediction)]
                                    # Backward only needs majority, not ridge fit; public data loading is shared.
                                    setup_cost=loading+setup['exact_export_seconds']+(fit.wall_seconds if arm.startswith(('forward','joint')) else 0.)
                                    cell={'cell_id':name,'execution_id':name,'measurement_reused':False,'victim_seed':seed,'depth':depth,'knowledge':knowledge,'attack_seed':attack_seed,'mode':mode,'arm':arm,
                                          'setup_id':setup_id,'work':asdict(work),'cold_seconds':setup_cost+work.wall_seconds,
                                          'amortized_seconds_per_document':work.wall_seconds/len(evaluation),
                                          'prediction_file':prediction_path.name,'prediction_sha256':file_sha256(prediction_path),'metrics':metrics}
                                    report['cells'].append(cell)
                                    ev.event('cell_complete',{k:v for k,v in cell.items() if k!='metrics'})
                                    if mode=='training' and arm.startswith(('forward_','sipit_')):
                                        measured_activation_cells[attack_seed,arm]=cell
                        del split,head,embedding,decoder
                # Defended-capture arms: a fresh defended victim per
                # (defense, depth) from the same pre-training base, trained on
                # the same frozen partition in the same shuffled order, with
                # the post-defense wires attacked under the identical access
                # spec as the undefended paired control. Any failure records
                # FAILED cells for every not-yet-completed arm of the block —
                # cells are never dropped — and blocks the run's COMPLETE
                # status below (the analyzer fails closed on it).
                for defense in scheduled_defenses:
                    for depth in p['depths']:
                        expected_block=[(knowledge,attack_seed,arm)
                                        for knowledge in p['knowledge'] for attack_seed in p['attack_seeds']
                                        for arm in defended_arms(p,defense)]
                        block_ids=[f'{seed}-{depth}-{knowledge}-{attack_seed}-training-{arm}'
                                   for knowledge,attack_seed,arm in expected_block]
                        if not any(name in owned for name in block_ids): continue  # block fully owned by other shards
                        completed=set()
                        try:
                            order=list(range(len(train))); random.Random(seed).shuffle(order)
                            train_batches=[train[i:i+1] for i in order]
                            block=run_defended_capture(args,p,base,train_batches,evaluation,seed,depth,defense)
                            report['defended_victims'].append(
                                {'seed':seed,'depth':depth,'defense':defense,
                                 'training_seconds':block['training_seconds'],
                                 'evaluation_nll':block['evaluation_nll'],
                                 'capture_dir':block['capture_dir'],
                                 'provenance':block['provenance'],'utility':block['utility']})
                            for knowledge in p['knowledge']:
                                if knowledge=='public':
                                    dhead_path=args.output/f'public-{seed}-{depth}.pt'  # identical grant as the control
                                    dexport_seconds=0.
                                else:
                                    dhead_path=args.output/f'defended-{defense}-{seed}-{depth}'/f'exact-{seed}-{depth}.pt'
                                    sync(); started=time.perf_counter()
                                    si.save_head_pt(str(dhead_path),block['victim'],depth)
                                    sync(); dexport_seconds=time.perf_counter()-started
                                sync(); started=time.perf_counter()
                                dsplit,dembedding=si._load_head_pt(str(dhead_path),args.device)
                                dsplit.model.requires_grad_(False); dembedding=dembedding.detach()
                                dhead=MeteredHead(dsplit)
                                with torch.no_grad(): dhead(ids=calibration[:1])
                                sync(); dloading=time.perf_counter()-started
                                with dhead.measure() as dfit, torch.no_grad():
                                    dhc=dhead(ids=calibration).reshape(-1,dembedding.shape[-1])
                                    ddecoder,_=si._ridge_fit(dhc,dembedding[calibration.flatten()],0)
                                dmajority=si.majority_control_token(calibration.tolist())
                                dsetup_id=f'{seed}-{depth}-{knowledge}@{defense}'
                                for attack_seed in p['attack_seeds']:
                                    dmethods=defended_arms(p,defense)
                                    random.Random(seed+attack_seed+depth).shuffle(dmethods)
                                    for arm in dmethods:
                                        name=f'{seed}-{depth}-{knowledge}-{attack_seed}-training-{arm}'
                                        if name not in owned: continue  # cell belongs to another shard
                                        with dhead.measure() as work:
                                            prediction,diagnostics=attack_observations(dhead,block['hidden'],None,
                                                dembedding,ddecoder,dmajority,arm,p,attack_seed)
                                        prediction_path=args.output/f'{name}.json'
                                        prediction_path.write_text(json.dumps({'predictions':prediction,'document_ids':[r['id'] for r in data['evaluation']], 'diagnostics':diagnostics},allow_nan=False))
                                        metrics=[{'document_id':row['id'], **reconstruction_metrics(row['tokens'],predicted,row.get('annotations',[]))}
                                                 for row,predicted in zip(data['evaluation'],prediction)]
                                        setup_cost=dloading+dexport_seconds+(dfit.wall_seconds if arm.startswith('forward') else 0.)
                                        base_arm=arm.split('@',1)[0]
                                        control_id=f'{seed}-{depth}-{knowledge}-{attack_seed}-training-{base_arm}'
                                        cell={'cell_id':name,'execution_id':name,'measurement_reused':False,'victim_seed':seed,'depth':depth,'knowledge':knowledge,'attack_seed':attack_seed,'mode':'training','arm':arm,
                                              'setup_id':dsetup_id,'work':asdict(work),'cold_seconds':setup_cost+work.wall_seconds,
                                              'amortized_seconds_per_document':work.wall_seconds/len(evaluation),
                                              'prediction_file':prediction_path.name,'prediction_sha256':file_sha256(prediction_path),'metrics':metrics,
                                              'status':'COMPLETE','defense':defense,'paired_control':control_id,
                                              'defense_provenance':block['provenance'],'defense_utility':block['utility'],
                                              'defended_observations':block['observations'],
                                              'access':{'knowledge':knowledge,'attack_seed':attack_seed,'budget_arm':base_arm,
                                                        'head_sha256':file_sha256(dhead_path),'calibration_documents':len(data['calibration']),
                                                        'observations':'post-defense wire captures of the same fixed evaluation documents',
                                                        'spec':'identical grant, budgets, calibration and seeds as the paired undefended control; only the observation surface is defended'}}
                                        report['cells'].append(cell)
                                        ev.event('cell_complete',{k:v for k,v in cell.items() if k!='metrics'})
                                        completed.add((knowledge,attack_seed,arm))
                                del dsplit,dhead,dembedding,ddecoder
                            del block
                        except Exception as error:
                            defended_failures.append(f'{defense}-{seed}-{depth}: {type(error).__name__}: {error}')
                            for knowledge,attack_seed,arm in expected_block:
                                if (knowledge,attack_seed,arm) in completed: continue
                                name=f'{seed}-{depth}-{knowledge}-{attack_seed}-training-{arm}'
                                if name not in owned: continue  # FAILED record is the owning shard's duty
                                cell={'cell_id':name,'execution_id':name,'measurement_reused':False,
                                      'victim_seed':seed,'depth':depth,'knowledge':knowledge,'attack_seed':attack_seed,
                                      'mode':'training','arm':arm,'defense':defense,'status':'FAILED',
                                      'error':f'{type(error).__name__}: {error}'}
                                report['cells'].append(cell)
                                ev.event('cell_failed',cell)
                save()
                print(f'Victim {seed} complete: {len(report["cells"])} logical cells',flush=True)
                del victim,optim,vs,hidden,gradient,hs,gs,h,base
            report['artifacts']={path.name:file_sha256(path) for path in args.output.glob('*.pt')}
            report['physical_execution_count']=sum(not c['measurement_reused'] for c in report['cells'])
            report['logical_cell_count']=len(report['cells'])
            if defended_failures:
                report['status']='FAILED'
                report['error']='defended-capture cells failed and were retained as FAILED: '+' | '.join(defended_failures)
                ev.event('failure',report['error'])
            else:
                report['status']='COMPLETE'
        except BaseException as error:
            report['status']='FAILED'; report['error']=f'{type(error).__name__}: {error}'; ev.event('failure',report['error']); raise
        finally: save()
    return 0


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',type=Path,default=Path('experiments/validation_protocol.json'))
    ap.add_argument('--input',type=Path); ap.add_argument('--input-sha256')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda'); ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--shard-count',type=int,default=1,
                    help='split the cell matrix into N deterministic shards (default 1: unchanged unsharded run)')
    ap.add_argument('--shard-index',type=int,default=0,
                    help='0-based index of the shard this process executes; requires 0 <= index < shard-count')
    return run(ap.parse_args())

if __name__=='__main__': raise SystemExit(main())
