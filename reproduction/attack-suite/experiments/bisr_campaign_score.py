"""Score only a complete paired campaign; never load models or make predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attacker.bisr_fidelity import sha256,score_predictions,compare_frozen_scores,write_predictions


def verified_rows(path,expected_ids,manifest_sha):
    finish=json.loads((path/'result.json').read_text())
    if finish['status']!='PREDICTIONS_COMPLETE' or finish['documents']!=len(expected_ids):
        raise ValueError('Incomplete prediction cell')
    if finish['manifest_sha256']!=manifest_sha:raise ValueError('Wrong prediction manifest')
    journal=[json.loads(x) for x in (path/'journal.jsonl').read_text().splitlines()]
    if len(journal)!=len(expected_ids):raise ValueError('Incomplete journal')
    previous='0'*64;rows=[]
    for i,(sample_id,event) in enumerate(zip(expected_ids,journal)):
        digest=event.pop('hash')
        if digest!=hashlib.sha256(json.dumps(event,sort_keys=True).encode()).hexdigest():
            raise ValueError('Journal hash mismatch')
        if event['index']!=i or event['sample_id']!=sample_id or event['previous']!=previous:
            raise ValueError('Journal chain mismatch')
        previous=digest;p=path/f'{i:05d}.json'
        if sha256(p)!=event['prediction_sha256']:raise ValueError('Prediction digest mismatch')
        row=json.loads(p.read_text())
        if row['sample_ids']!=[sample_id] or row['input_manifest_sha256']!=manifest_sha:
            raise ValueError('Prediction identities mismatch')
        rows.append(row)
    if previous!=finish['journal_tip']:raise ValueError('Wrong journal tip')
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('campaign','inputs','tokenizer','output'):ap.add_argument('--'+n,type=Path,required=True)
    args=ap.parse_args()
    from transformers import AutoTokenizer
    frozen=json.loads((args.inputs/'manifest.json').read_text())
    if sha256(args.inputs/'truth.json')!=frozen['outputs']['truth.json']:raise ValueError('Truth changed')
    truth=json.loads((args.inputs/'truth.json').read_text());ids=frozen['validation_ids']
    if truth['sample_ids']!=ids:raise ValueError('Truth identity mismatch')
    bundle=json.loads((args.campaign/'records/launch.json').read_text())
    if bundle['input_manifest_sha256']!=sha256(args.inputs/'manifest.json'):raise ValueError('Wrong frozen set')
    expected={f's{s}-d{d}-{g}' for s in (31415,27182,16180) for d in (3,6) for g in ('public','exact')}
    if set(bundle['cells'])!=expected:raise ValueError('Incomplete prespecified cell set')
    # Prove completeness/integrity for every cell before looking at any recovery.
    complete={}
    for key,cell in bundle['cells'].items():
        for engine in ('candidate','reference'):
            complete[key,engine]=verified_rows(args.campaign/'predictions'/key/engine,ids,cell['manifest_sha256'])
    tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    args.output.mkdir(parents=True,exist_ok=False);scores={};results={}
    for (key,engine),records in complete.items():
        merged={'schema':'bisr.predictions.v1','sample_ids':ids,
                'arms':{a:[r['arms'][a][0] for r in records] for a in ('sip','b','f','bf')}}
        scores[key,engine]=score_predictions(merged,truth,tok)
        write_predictions(args.output/f'{key}-{engine}.json',scores[key,engine])
    for depth in (3,6):
        for grant in ('public','exact'):
            paired={engine:{str(s):scores[f's{s}-d{depth}-{grant}',engine] for s in (31415,27182,16180)}
                    for engine in ('candidate','reference')}
            results[f'd{depth}-{grant}']=compare_frozen_scores(paired['candidate'],paired['reference'],frozen)
    statuses=[r['statistical_status'] for r in results.values()]
    verdict='CONFORMANCE_ESTABLISHED_FOR_DECLARED_C_A' if all(s=='EQUIVALENT' for s in statuses) else (
        'NON_EQUIVALENT' if 'NON_EQUIVALENT' in statuses else 'INCONCLUSIVE')
    write_predictions(args.output/'result.json',{'status':verdict,'strata':results,
        'scope':'single-seam GPT-2 implementation conformance on fixed fresh WikiText-103 holdout',
        'published_setting_reproduced':False,'input_manifest_sha256':sha256(args.inputs/'manifest.json'),
        'launch_manifest_sha256':sha256(args.campaign/'records/launch.json')})


if __name__=='__main__':main()
