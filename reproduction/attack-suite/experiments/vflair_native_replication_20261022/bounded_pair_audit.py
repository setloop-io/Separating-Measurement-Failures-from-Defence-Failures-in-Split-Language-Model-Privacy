"""Standard-library audit of a completed native MID/NONE pair.

This reads frozen metadata, IDs and artifact bytes. It never imports a model,
deserializes a pickle/tensor, executes an attack, or assigns a protection label.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random

from bounded_study import assert_launch_authorized,validate_inputs,validate_native_counts

ATTACKS = {'VanillaModelInversion_WhiteBox':(400,.001), 'WhiteBoxInversion':(300,.005)}
UPSTREAM = 'ff18565d89dce6367b6f13881a415c053b60cb0f'
MANIFEST = '49fc5ac15586bf35f44acdfb02cb8830671da6cdb6474b4723a4907cdd3fe447'
CRITICAL_CODE = ['native_pipeline.py','runner.py','dispatch.py','verify_run.py',
                 'bounded_study.py','import_compat.py','post_run_verify.py']


def _require(condition,message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _asset_index(out,name):
    entries = _json(out/name)
    _require(isinstance(entries,list) and bool(entries),f'{name}: missing artifact entries')
    paths = set()
    for entry in entries:
        relative = Path(entry['path'])
        path = (out/relative).resolve()
        _require(not relative.is_absolute() and path.is_relative_to(out),f'{name}: artifact path escaped run')
        _require(str(relative) not in paths,f'{name}: duplicate artifact path')
        paths.add(str(relative))
        _require(path.is_file() and path.stat().st_size==entry['bytes'] and _sha(path)==entry['sha256'],f'{name}: artifact hash/size changed: {relative}')
    return entries


def _read_run(out,condition):
    out = Path(out).resolve()
    cfg = _json(out/'effective-run.json'); assert_launch_authorized(cfg)
    _require(cfg['phase']=='bounded-study' and cfg['condition']==condition,'Wrong bounded-study condition')
    terminal = _json(out/'terminal.json'); summary = _json(out/'native-summary.json')
    _require(terminal['status']=='complete' and terminal['exit_code']==0 and summary['status']=='complete','Run is incomplete or timed out')
    verification = _json(out/'verification.json')
    _require(verification.get('status')=='PASS' and verification.get('phase')=='bounded-study' and verification.get('condition')==condition,'Retained-output verification did not pass')
    _require(verification.get('utility_rows_verified')==5153 and verification.get('prediction_rows_verified')==400 and verification.get('capture_files_verified')==400,'Retained-output verification has incomplete coverage')
    _require(summary['condition']==condition and summary['seed']==cfg['seed'],'Summary seed/condition mismatch')
    inputs = _json(out/'inputs.json'); receipt = validate_inputs(cfg,inputs)
    source = _json(out/'upstream-verification.json')
    _require(source['revision']==UPSTREAM and source['manifest_sha256']==MANIFEST and len(source['files'])==254,'Pinned upstream source manifest changed')
    _require(len({x['path'] for x in source['files']})==254,'Duplicated upstream file entries')
    compat = _json(out/'compatibility.json')
    _require(compat['profile']=='gpt2-imports-v1' and all(x['function_class_ast_preserved'] for x in compat['changes']),'Native scientific function/class preservation failed')
    selection = _json(out/'bounded-study-selection-receipt.json')
    _require(selection['source_indices']==receipt['source_indices'] and selection['all_test_rows_retained']==5153,'Fixed-book/test selection changed')
    _require(selection['selection_receipt_sha256']==cfg['bounded_study']['selection_receipt_sha256'] and selection['proposal_sha256']==receipt['proposal']['sha256'],'Selection/protocol receipt binding changed')
    _require(_sha(out/'bounded-study-selection-receipt.json')==summary['bounded_study_selection_receipt_sha256'],'Execution selection receipt hash changed')
    books = selection['selected_books']
    _require([{k:x[k] for k in ['split','index','tokens','native_windows']} for x in books]==receipt['selected_books'],'Selected book/window metadata changed')
    _require(all(len(x['row_text_domain_sha256'])==64 and len(x['text_utf8_sha256'])==64 for x in books),'Selected row hashes missing')
    counts = _json(out/'native-data-counts.json'); validate_native_counts(cfg,counts)
    native = _json(out/'native-effective-config.json')
    _require(native['epochs']==30 and native['batch_size']==16 and native['early_stop_threshold']==1,'Native training limits changed')
    final_epoch = summary['final_epoch']
    _require(type(final_epoch) is int and 1<=final_epoch<=30,'Invalid native final epoch')
    steps = _lines(out/'steps.jsonl')
    _require(len(steps)==104*final_epoch==summary['native_training_steps']<=3120,'Incomplete native epoch/update endpoint')
    _require(all(x['step']==i+1 and x['epoch']==i//104 and x.get('loss_finite') is True and math.isfinite(float(x['loss'])) and x.get('training_autocast','none')=='none' for i,x in enumerate(steps)),'Native step trajectory or FP32/finite-loss condition failed')
    defense = _json(out/'native-defense-runtime.json')
    _require(defense['condition']==condition and defense['victim_seed']==cfg['seed'],'Runtime defense receipt seed/condition changed')
    if condition=='mid':
        _require(defense['apply_mid'] is True and defense['mid_class']=='MIDModel_Linear' and defense['mid_std_shift']==5,'Native MID runtime differs from effective std_shift5')
    else:
        _require(defense['apply_mid'] is False and defense['mid_class'] is None and defense['mid_std_shift'] is None,'NONE contains a learned MID defense')
    utility = _json(out/'utility-predictions.json')
    _require(len(utility)==1,'Expected exactly one complete native utility pass')
    utility = utility[0]
    _require(len(utility['target_ids'])==len(utility['predicted_ids'])==5153,'Native utility rows incomplete')
    accuracy = sum(p==t for p,t in zip(utility['predicted_ids'],utility['target_ids']))/5153
    _require(abs(accuracy-utility['native_result']['acc'])<1e-12 and abs(accuracy-summary['native_utility'])<1e-12,'Native utility score disagrees with frozen IDs')
    captures = _asset_index(out,'capture-index.json'); _asset_index(out,'checkpoint-index.json')
    _require(len(captures)==400 and [x['index'] for x in captures]==list(range(400)),'Native capture count/ordering incomplete')
    predictions = _lines(out/'attack-predictions.jsonl')
    _require(len(predictions)==400,'Native prediction coverage incomplete')
    config_attacks = {v['name']:v['parameters'] for v in native['attack_list'].values()}
    summaries = _json(out/'attack-summaries.json')
    _require(set(config_attacks)==set(ATTACKS) and len(summaries)==2 and {x['attack'] for x in summaries}==set(ATTACKS),'Native attack set changed')
    _require(summaries==summary['attack_summaries'],'Native attack summary copies disagree')
    routes = {}; used_captures = set()
    for attack in summaries:
        name = attack['attack']; params = config_attacks[name]; iterations,lr = ATTACKS[name]
        _require(params['epochs']==iterations and params['lr']==lr and params['attack_sample_num']==200 and params['batch_size']==1 and params['target_data']=='test' and params.get('loss_type','cross_entropy')=='cross_entropy','Native attack budget/objective changed')
        if name=='WhiteBoxInversion': _require(params['T']==.05,'Native WBI temperature changed')
        attack_seed = params.get('attack_seed',cfg['seed'])
        _require(type(attack_seed) is int,'Invalid native attack seed')
        expected = random.Random(attack_seed).sample(range(5153),200)
        calls = [x for x in attack['sample_selection_calls'] if x.get('population_size')==5153 and x.get('k')==200]
        _require(len(calls)==1 and calls[0]['selected']==expected,'Native target selection differs from its victim/explicit attack seed')
        rows = [x for x in predictions if x['attack']==name]
        _require(len(rows)==attack['prediction_rows']==200 and [x['sample_ordinal'] for x in rows]==list(range(200)),'Attack target rows/order incomplete')
        for row in rows:
            ordinal = row['capture_ordinal']
            _require(type(ordinal) is int and 0<=ordinal<400 and ordinal not in used_captures,'Invalid/reused capture ordinal')
            used_captures.add(ordinal)
            _require(captures[ordinal]['attack']==name and captures[ordinal]['index']==ordinal,'Prediction/capture route linkage changed')
            truth = [x for x in row['original_ids'] if x!=row['pad_token_id']]; predicted = row['predicted_ids']
            _require(bool(truth) and bool(predicted),'Empty attack ID sequence')
            precision = sum(x in truth for x in predicted)/len(predicted); recall = sum(x in predicted for x in truth)/len(truth)
            _require(abs(precision-row['native_membership_precision_recomputed'])<1e-12 and abs(recall-row['native_membership_recall_recomputed'])<1e-12,'Attack score disagrees with frozen IDs')
        mean_precision = sum(x['native_membership_precision_recomputed'] for x in rows)/200
        mean_recall = sum(x['native_membership_recall_recomputed'] for x in rows)/200
        _require(abs(mean_precision-attack['native_return'][0])<1e-10 and abs(mean_recall-attack['native_return'][1])<1e-10,'Native attack return disagrees with frozen IDs')
        routes[name] = {'attack_seed':attack_seed,'target_indices':expected,'original_ids':[x['original_ids'] for x in rows],
                        'ordered_target_sha256':hashlib.sha256(json.dumps(expected,separators=(',',':')).encode()).hexdigest()}
    _require(len(used_captures)==400,'Some native captures have no prediction')
    names = list(ATTACKS); by_index = [{index:ids for index,ids in zip(routes[n]['target_indices'],routes[n]['original_ids'])} for n in names]
    shared = set(by_index[0]) & set(by_index[1])
    _require(all(by_index[0][i]==by_index[1][i] for i in shared),'VMI/WBI original IDs disagree for a shared corpus target')
    initial = {(x['party'],x['name']):x for x in _json(out/'native-trainability.json')}
    for key in [(0,'local_model'),(1,'global_model')]:
        _require(key in initial and initial[key]['parameter_dtypes']==['torch.float32'],'Native head/tail initial precision changed')
    provenance = _json(out/'launcher-provenance.json')
    _require(all(k in provenance['task_files'] for k in CRITICAL_CODE),'Frozen critical source hashes missing')
    evidence = [p for p in out.iterdir() if p.is_file() and p.suffix in ['.json','.jsonl']]
    return {'out':out,'cfg':cfg,'source':source,'compat':compat,'selection':selection,'utility':utility,
            'native':native,'routes':routes,'initial':initial,'critical_code':{k:provenance['task_files'][k] for k in CRITICAL_CODE},
            'evidence_files':{p.name:_sha(p) for p in evidence},'same_route_targets':routes[names[0]]['target_indices']==routes[names[1]]['target_indices']}


def verify_pair(mid_out,none_out,receipt_out):
    """Return/write VERIFIED evidence receipt, or raise ValueError on mismatch."""
    mid = _read_run(mid_out,'mid'); none = _read_run(none_out,'none')
    _require(mid['cfg']['seed']==none['cfg']['seed'],'Paired victim seeds differ')
    _require(mid['source']==none['source'] and mid['compat']==none['compat'],'Paired upstream/import compatibility source differs')
    _require(mid['critical_code']==none['critical_code'],'Paired frozen execution/verification source differs')
    _require(mid['selection']['selected_books']==none['selection']['selected_books'] and mid['selection']['source_indices']==none['selection']['source_indices'],'Paired source-book row identities differ')
    _require(mid['utility']['target_ids']==none['utility']['target_ids'],'Paired held-out utility targets differ')
    for key in [(0,'local_model'),(1,'global_model')]:
        _require(mid['initial'][key]['state_sha256']==none['initial'][key]['state_sha256'],'Paired initial native head/tail checkpoint differs')
    normalized = []
    for run in [mid,none]:
        config = copy.deepcopy(run['native']); config.pop('defense',None)
        for party in ['0','1']: config['model_list'][party]['path']='__RUN_MODEL_PATH__'
        normalized.append(config)
    _require(normalized[0]==normalized[1],'Paired native configurations differ beyond defense/run model paths')
    for name in ATTACKS:
        _require(mid['routes'][name]==none['routes'][name],f'Paired {name} attack seeds, sampled targets, or original ID sequences differ')
    result = {'status':'VERIFIED','audit_scope':'frozen evidence consistency; no categorical protection/PASS interpretation',
              'victim_seed':mid['cfg']['seed'],'mid_out':str(mid['out']),'none_out':str(none['out']),
              'study_id':mid['cfg']['study_id'],'utility_rows_per_condition':5153,'attack_targets_per_route_per_condition':200,
              'capture_files_per_condition':400,'native_mid_class':'MIDModel_Linear','native_mid_effective_std_shift':5,
              'paired_source_model_dataset_and_selected_row_pins':True,'paired_initial_head_tail_states':True,
              'paired_ordered_targets_and_original_ids':True,
              'same_targets_across_routes':mid['same_route_targets'] and none['same_route_targets'],
              'target_receipts':{name:{k:v for k,v in route.items() if k!='original_ids'} for name,route in mid['routes'].items()},
              'source_files':mid['critical_code'],'evidence_hashes':{'mid':mid['evidence_files'],'none':none['evidence_files']},
              'model_attack_or_pickle_load_executed_by_auditor':False}
    path = Path(receipt_out); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mid-out',required=True); parser.add_argument('--none-out',required=True); parser.add_argument('--receipt-out',required=True)
    args = parser.parse_args()
    print(json.dumps(verify_pair(args.mid_out,args.none_out,args.receipt_out),indent=2))
