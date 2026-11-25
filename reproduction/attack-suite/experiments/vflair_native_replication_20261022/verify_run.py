"""CPU-only verification of a terminal native run; never launches a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def verify(out):
    import torch
    def read(name):
        return json.loads((out/name).read_text())
    def jsonl(name):
        return [json.loads(x) for x in (out/name).read_text().splitlines() if x.strip()]
    def require(condition,message):
        if not condition:
            raise RuntimeError(message)
    terminal = read('terminal.json')
    require(terminal['status']=='complete' and terminal['exit_code']==0,'Run is not terminal-successful')
    summary,cfg = read('native-summary.json'),read('effective-run.json')
    require(summary['status']=='complete','Native pipeline did not complete')
    require(read('admission.json')['status']=='ADMITTED','Inputs/imports were not admitted')
    compat = read('compatibility.json')
    require(all(x['function_class_ast_preserved'] for x in compat['changes']),'Native class/function body preservation failed')
    steps = jsonl('steps.jsonl')
    counts = read('native-data-counts.json')
    if cfg['phase']=='bounded-study':
        from bounded_study import SOURCE_INDICES,validate_native_counts
        validate_native_counts(cfg,counts)
        selected = read('bounded-study-selection-receipt.json')
        require(selected['source_indices']==SOURCE_INDICES,'Bounded-study source selection changed')
        require(selected['all_test_rows_retained']==5153,'Bounded-study utility split changed')
        require(selected['selection_receipt_sha256']==cfg['bounded_study']['selection_receipt_sha256'],'Bounded-study selection binding changed')
        require(sha(out/'bounded-study-selection-receipt.json')==summary['bounded_study_selection_receipt_sha256'],'Bounded-study execution selection receipt hash changed')
        defense = read('native-defense-runtime.json')
        require(defense['condition']==cfg['condition'] and defense['victim_seed']==cfg['seed'],'Native defense receipt identity changed')
        if cfg['condition']=='mid':
            require(defense['apply_mid'] and defense['mid_class']=='MIDModel_Linear' and defense['mid_std_shift']==5,'Effective native MID implementation/defaults changed')
        else:
            require(not defense['apply_mid'] and defense['mid_class'] is None,'Matched NONE unexpectedly contains MID')
    native_epochs = read('native-effective-config.json')['epochs']
    expected_steps = 1 if native_epochs==1 else counts['native_batches_per_epoch']*summary['final_epoch']
    require(len(steps)==summary['native_training_steps']==expected_steps,'Native optimizer trajectory is incomplete')
    require([x['step'] for x in steps]==list(range(1,len(steps)+1)),'Noncontiguous optimizer step receipts')
    require(all(math.isfinite(float(x['loss'])) for x in steps),'Nonfinite retained training loss')
    initial,final = read('native-trainability.json'),read('native-final-weights.json')
    before = {(x['party'],x['name']):x for x in initial}
    changed = []
    for x in final:
        prev = before[(x['party'],x['name'])]
        if x['state_sha256']!=prev['state_sha256']:
            changed.append({'party':x['party'],'name':x['name']})
    require({'party':0,'name':'local_model'} in changed,'Native trainable head did not change')
    require({'party':1,'name':'global_model'} in changed,'Native trainable tail did not change')
    if cfg['phase']=='bounded-study' and cfg['condition']=='mid':
        require({'party':0,'name':'head_mid_model'} in changed,'Native learned MID component did not change')
    hashes = []
    for indexname in ['checkpoint-index.json','capture-index.json']:
        for item in read(indexname):
            p = (out/item['path']).resolve()
            require(p.is_relative_to(out),'Artifact path escaped run directory')
            require(p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],f'Artifact hash mismatch: {p}')
            hashes.append({'path':item['path'],'sha256':item['sha256']})
    captures = read('capture-index.json')
    predictions = jsonl('attack-predictions.jsonl')
    if cfg['phase']=='bounded-study':
        require(len(predictions)==len(captures)==400,'Bounded study requires all 400 predictions and actual wire captures')
    grouped = {}
    for row in predictions:
        original,predicted,pad = row['original_ids'],row['predicted_ids'],row['pad_token_id']
        clean = [x for x in original if x!=pad]
        precision = sum(x in clean for x in predicted)/len(predicted)
        recall = sum(x in predicted for x in clean)/len(clean)
        require(abs(precision-row['native_membership_precision_recomputed'])<1e-12,'Stored precision mismatch')
        require(abs(recall-row['native_membership_recall_recomputed'])<1e-12,'Stored recall mismatch')
        cap = captures[row['capture_ordinal']]
        require(cap['attack']==row['attack'],'Prediction/capture attack mismatch')
        payload = torch.load(out/cap['path'],map_location='cpu',weights_only=True)
        native_ids = payload['native_local_data_input']['input_ids'].reshape(-1).tolist()
        require(native_ids==original,'Prediction truth IDs disagree with the captured native input')
        active = [i for i,x in enumerate(original) if x!=pad]
        if len(predicted)==len(original):
            matches = sum(original[i]==predicted[i] for i in active)
            require(matches==row['position_matches_excluding_pad'],'Positional matches mismatch')
        grouped.setdefault(row['attack'],[]).append(row)
    for attack in read('attack-summaries.json'):
        rows = grouped.get(attack['attack'],[])
        require(len(rows)==attack['prediction_rows'],'Attack coverage receipt mismatch')
        require([x['sample_ordinal'] for x in rows]==list(range(len(rows))),'Attack sample order incomplete')
        precision = sum(x['native_membership_precision_recomputed'] for x in rows)/len(rows)
        recall = sum(x['native_membership_recall_recomputed'] for x in rows)/len(rows)
        require(abs(precision-attack['native_return'][0])<1e-10 and abs(recall-attack['native_return'][1])<1e-10,'Native return disagrees with independent scores')
    utility = read('utility-predictions.json')
    if cfg['phase']=='bounded-study':
        attacks = read('attack-summaries.json')
        require(len(attacks)==2 and {x['attack'] for x in attacks}=={'VanillaModelInversion_WhiteBox','WhiteBoxInversion'},'Bounded attack-family coverage changed')
        require(all(x['prediction_rows']==200 for x in attacks),'Bounded study requires 200 targets per attack')
        require(len(utility)==1,'Bounded study requires one complete native utility evaluation')
    for row in utility:
        predicted,target = row['predicted_ids'],row['target_ids']
        require(len(predicted)==len(target)==counts['test_rows'],'Utility coverage is incomplete')
        if cfg.get('utility_metric','LAMBADA').startswith('LAMBADA') or cfg.get('utility_metric') is None:
            accuracy = sum(p==t for p,t in zip(predicted,target))/len(target)
            require(abs(accuracy-row['native_result']['acc'])<1e-12,'Utility score disagrees with retained IDs')
    result = {'status':'PASS','model_or_attack_executed_by_verifier':False,'phase':cfg['phase'],
              'condition':cfg['condition'],'native_steps':len(steps),'native_components_changed':changed,
              'prediction_rows_verified':len(predictions),'capture_files_verified':len(captures),
              'utility_rows_verified':counts['test_rows'],'artifact_hashes_verified':len(hashes),
              'native_source_functions_classes_preserved':True,
              'optimizer_rng_sha256':sha(out/'native-final-optimizer-rng.pt')}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    a = p.parse_args()
    print(json.dumps(verify(Path(a.output).resolve()),indent=2))
