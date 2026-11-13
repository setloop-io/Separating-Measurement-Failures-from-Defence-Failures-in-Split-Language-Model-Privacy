"""Fixed-book native study admission and selection; no models or compute launch.

The shipped profiles are blocked. A later user scope/allocation choice must be
recorded in a new frozen child configuration before any execution is permitted.
"""
import hashlib
import json
import math
from pathlib import Path
import random

PHASE = 'bounded-study'
STUDY_ID = 'native-mid-gpt2-lambada-fp32-fixed8books-20261022'
SOURCE_INDICES = [198,706,943,960,989,1158,1444,2372]
AUTHORIZED = 'AUTHORIZED_TO_LAUNCH'


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def is_bounded_profile(cfg):
    # The persistent marker also prevents --phase full/canary from bypassing
    # the authorization gate on a prepared bounded-study profile.
    return cfg.get('phase')==PHASE or cfg.get('study_id')==STUDY_ID or 'bounded_study' in cfg


def load_selection(cfg):
    binding = cfg.get('bounded_study',{})
    if binding.get('selection_receipt')!='bounded-study-selection.json':
        raise ValueError('Bounded study requires the dedicated frozen selection receipt')
    path = Path(__file__).with_name('bounded-study-selection.json')
    if sha256_file(path)!=binding.get('selection_receipt_sha256'):
        raise ValueError('Bounded-study selection receipt hash mismatch')
    receipt = json.loads(path.read_text())
    if binding.get('protocol_sha256')!=receipt['proposal']['sha256']:
        raise ValueError('Bounded-study proposal hash binding mismatch')
    if receipt['source_indices']!=SOURCE_INDICES or sorted(random.Random(20261022).sample(range(2662),8))!=SOURCE_INDICES:
        raise ValueError('Bounded-study uniform fixed selection changed')
    return receipt


def validate_configuration(cfg):
    if cfg.get('phase')!=PHASE or cfg.get('study_id')!=STUDY_ID:
        raise ValueError('Bounded-study marker requires its dedicated phase and study ID')
    receipt = load_selection(cfg)
    if cfg.get('condition') not in ['mid','none']:
        raise ValueError('Bounded study contains native MID and matched NONE only')
    if type(cfg.get('seed')) is not int or cfg['seed'] not in [11,22,33,60]:
        raise ValueError('Bounded-study seed is outside the frozen three-seed/default-seed options')
    if cfg.get('training_autocast','none')!='none':
        raise ValueError('Bounded study preserves the original FP32 training path')
    expected = {'epochs':30,'batch_size':16,'sequence_length':512,'learning_rate':0.001,
                'native_early_stop_threshold':1,'include_bisr':False}
    if any(cfg.get('full',{}).get(k)!=v for k,v in expected.items()):
        raise ValueError('Bounded-study native configuration/attack coverage changed')
    guards = {'maximum_training_windows':1655,'maximum_test_rows':5153,'maximum_optimizer_steps':3120}
    if any(cfg.get('guards',{}).get(k)!=v for k,v in guards.items()):
        raise ValueError('Bounded-study exact data/optimizer guard changed')
    if cfg.get('upstream_revision')!=receipt['upstream_revision']:
        raise ValueError('Bounded-study upstream revision changed')
    return receipt


def assert_launch_authorized(cfg):
    if not is_bounded_profile(cfg):
        return
    authorization = cfg.get('authorization',{})
    if cfg.get('launch_status')!=AUTHORIZED or authorization.get('status')!=AUTHORIZED:
        raise PermissionError('NOT_AUTHORIZED_TO_LAUNCH: the bounded native study is prepared only; user scope/allocation choice is pending')
    validate_configuration(cfg)
    for field in ['user_scope_choice','allocation_reference']:
        if not isinstance(authorization.get(field),str) or not authorization[field].strip():
            raise PermissionError(f'Bounded-study authorization requires a recorded {field}')
    wall = cfg.get('guards',{}).get('wall_seconds')
    if type(wall) is not int or wall<=0 or authorization.get('per_run_wall_seconds')!=wall:
        raise PermissionError('Bounded-study execution requires an explicitly approved positive per-run wall guard')
    aggregate = authorization.get('approved_aggregate_gpu_hours')
    if isinstance(aggregate,bool) or not isinstance(aggregate,(int,float)) or not math.isfinite(aggregate) or aggregate<=0:
        raise PermissionError('Bounded-study execution requires a fresh aggregate allocation reference; no older budget is inherited')
    if wall>aggregate*3600:
        raise PermissionError('Per-run wall guard exceeds the recorded aggregate allocation')


def validate_original_config(cfg,path):
    receipt = validate_configuration(cfg)
    if sha256_file(path)!=receipt['original_mid_config_sha256']:
        raise ValueError('Bounded-study original native MID configuration bytes changed')


def validate_inputs(cfg,inputs):
    receipt = validate_configuration(cfg)
    if inputs['dataset']['revision']!=receipt['dataset_revision']:
        raise ValueError('Bounded-study dataset revision changed')
    for split in ['train','test']:
        actual = [(x['sha256'],x['bytes'],x['rows']) for x in inputs['dataset']['splits'].get(split,[])]
        expected = [(x['sha256'],x['bytes'],x['rows']) for x in receipt['dataset_files'][split]]
        if actual!=expected:
            raise ValueError(f'Bounded-study {split} file identity/order/count changed')
    if inputs['model']['revision']!=receipt['model_revision']:
        raise ValueError('Bounded-study model revision changed')
    actual = {x['path']:(x['sha256'],x['bytes']) for x in inputs['model']['files']}
    for item in receipt['model_files']:
        if actual.get(item['path'])!=(item['sha256'],item['bytes']):
            raise ValueError(f'Bounded-study model/tokenizer asset changed: {item["path"]}')
    return receipt


def select_fixed_books(dataset,cfg,inputs):
    receipt = validate_inputs(cfg,inputs)
    if len(dataset['train'])!=2662 or len(dataset['test'])!=5153:
        raise ValueError('Bounded study requires complete pinned source splits before selection')
    selected_rows = []
    for expected in receipt['selected_books']:
        row = dataset['train'][expected['index']]
        payload = {'text':row['text'],'domain':row['domain']}
        canonical = json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
        selected_rows.append({**expected,'row_text_domain_sha256':hashlib.sha256(canonical).hexdigest(),
                              'text_utf8_sha256':hashlib.sha256(row['text'].encode('utf-8')).hexdigest()})
    selected = dict(dataset)
    selected['train'] = dataset['train'].select(SOURCE_INDICES)
    # The exact test dataset object is retained: no test row filtering/reordering.
    if selected['test'] is not dataset['test']:
        raise RuntimeError('Bounded-study test split identity changed')
    record = {'study_id':STUDY_ID,'phase':PHASE,'source_indices':SOURCE_INDICES,
              'draw_order':receipt['draw_order'],'uniform_selection_seed':20261022,
              'selection_receipt_sha256':cfg['bounded_study']['selection_receipt_sha256'],
              'proposal_sha256':receipt['proposal']['sha256'],'original_source_train_rows':2662,
              'selected_train_source_rows':8,'all_test_rows_retained':5153,
              'native_training_windows_expected':1655,'selected_books':selected_rows,
              'source_dataset_manifest_sha256':inputs['dataset']['manifest_sha256'],
              'data_adaptation':'exact fixed training-book subset; original native preprocessing unchanged',
              'quality_length_domain_or_outcome_filter':False,'test_split_object_preserved':True}
    return selected,record


def validate_native_counts(cfg,counts):
    validate_configuration(cfg)
    expected = {'train_windows':1655,'test_rows':5153,'native_batches_per_epoch':104,'native_max_training_steps':3120}
    if any(counts.get(k)!=v for k,v in expected.items()):
        raise ValueError(f'Bounded-study actual native data/loop counts differ: {counts}')
    return {'status':'PASS','study_id':STUDY_ID,'exact_native_counts':expected,
            'checkpoint_policy':'native final checkpoint after stopping/cap only; no exact interrupted resume',
            'aggregate_allocation_enforcement':'parent ORX scheduling; this child enforces only its approved per-run wall/step guards'}
