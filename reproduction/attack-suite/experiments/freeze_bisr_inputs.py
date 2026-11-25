"""Freeze amendment-v2 article selections and token windows; never run attacks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.prepare_pilot import articles


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def normal(text): return ' '.join(text.split())
def text_hash(text): return hashlib.sha256(normal(text).encode()).hexdigest()
def jsonl(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def main():
    import pyarrow as pa
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('arrow-dir','v3','pilot','old-aux','sensi','b1-local-corpus','b1-aux','tokenizer','amendment','output'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {k: getattr(args,k) for k in ('v3','pilot','old_aux','sensi','b1_local_corpus','b1_aux','amendment')}
    pins = {k:{'path':str(p), 'sha256':sha(p)} for k,p in inputs.items()}
    v3 = json.loads(args.v3.read_text())
    v3_ids = {s for values in v3['selected_ids'].values() for s in values}
    pilot, old_aux, b1_aux = jsonl(args.pilot), jsonl(args.old_aux), jsonl(args.b1_aux)
    forbidden_ids = v3_ids | {d['id'] for d in pilot}
    historical_hashes = {text_hash(d['text']) for d in pilot + old_aux + b1_aux}
    # Both raw columns are checked even when the historical substitute is identical.
    with args.sensi.open() as f:
        for row in csv.DictReader(f):
            for key in ('content', 'sani_gpt4'):
                if key not in row: raise ValueError('Missing historical Sensi column')
                historical_hashes.add(text_hash(row[key]))
    partitions = {}
    for split in ('train','test'):
        path = args.arrow_dir/f'wikitext-{split}.arrow'
        pins[split] = {'path':str(path), 'sha256':sha(path)}
        rows = pa.ipc.open_stream(path).read_all()['text'].to_pylist()
        partitions[split] = [{'id':f'wikitext2-{split}-'+hashlib.sha256(text.encode()).hexdigest()[:20],
                              'title':title, 'text':text, 'normalised_sha256':text_hash(text)}
                             for title,text in articles(rows)]
    train = partitions['train']
    # Whole source partition exclusion is conservative when B1 IDs are unavailable.
    historical_hashes.update(d['normalised_sha256'] for d in train)
    historical_titles = {normal(d['title']).casefold() for d in train}
    historical_hashes.update(text_hash(text) for _,text in articles(args.b1_local_corpus.read_text().splitlines()))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer_pins = {p.name:sha(p) for p in sorted(args.tokenizer.iterdir())
                      if p.is_file() and p.suffix in ('.json','.txt')}
    excluded=[]; validation=[]; seen=set()
    ordered = sorted(partitions['test'],key=lambda d:hashlib.sha256(('bisr-conformance-v1'+normal(d['text'])).encode()).hexdigest())
    for d in ordered:
        h=d['normalised_sha256']; reason=None
        ids=tokenizer.encode(d['text'],add_special_tokens=False)
        if h in historical_hashes or normal(d['title']).casefold() in historical_titles:
            reason='historical_document_or_title'
        elif h in seen: reason='duplicate_normalised_text'
        elif len(ids)<32: reason='shorter_than_32_tokens'
        if reason: excluded.append({'id':d['id'],'reason':reason});continue
        seen.add(h)
        validation.append({**d,'token_ids':ids[:32],'mask':[1]*32})
    if not validation: raise ValueError('No eligible validation articles')
    reserved={d['normalised_sha256'] for d in validation}
    pilot_hashes={text_hash(d['text']) for d in pilot}
    eligible=[d for d in train if d['id'] not in forbidden_ids and d['normalised_sha256'] not in reserved|pilot_hashes]
    # Check manifest identity really resolves to the pinned source partition.
    if not v3_ids.issubset({d['id'] for d in train}):
        raise ValueError('v3 IDs do not resolve to this pinned training partition')
    eligible.sort(key=lambda d:hashlib.sha256(('bisr-aux-v1'+normal(d['text'])).encode()).hexdigest())
    if len(eligible)<112: raise ValueError(f'Only {len(eligible)} auxiliary articles remain')
    aux=eligible[:112]; windows=[]; counts={}
    for d in aux:
        ids=tokenizer.encode(d['text'],add_special_tokens=False)
        counts[d['id']]=len(ids)//32
        for offset in range(0,len(ids)-31,32):
            windows.append({'sample_id':d['id']+f':{offset}', 'article_id':d['id'],
                            'token_ids':ids[offset:offset+32], 'mask':[1]*32})
    if not windows: raise ValueError('No complete auxiliary windows')
    files={'validation-documents.json':validation,'auxiliary-documents.json':aux,
           'auxiliary-windows.json':windows,'validation-exclusions.json':excluded,
           'truth.json':{'schema':'bisr.truth.v1','sample_ids':[d['id'] for d in validation],
                         'token_ids':[d['token_ids'] for d in validation]}}
    for name,value in files.items():
        (args.output/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    manifest={'schema':'bisr.frozen_inputs.v2','status':'INPUTS_FROZEN_CHECKPOINTS_NOT_ADMITTED',
      'script_sha256':sha(__file__),'source_pins':pins,'tokenizer_files':tokenizer_pins,
      'validation_documents':len(validation),'validation_ids':[d['id'] for d in validation],
      'auxiliary_articles':len(aux),'auxiliary_windows':len(windows),'auxiliary_windows_by_article':counts,
      'reused_historical_auxiliary_articles':sum(d['normalised_sha256'] in {text_hash(x['text']) for x in old_aux} for d in aux),
      'fresh_auxiliary_claim':False,'excluded_validation_documents':excluded,
      'historical_B1_policy':'exclude complete training partition from validation; auxiliary reuse permitted by v2',
      'outputs':{name:sha(args.output/name) for name in files},'launch_admitted':False}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('status','validation_documents','auxiliary_articles','auxiliary_windows','reused_historical_auxiliary_articles')}))


if __name__=='__main__': main()
