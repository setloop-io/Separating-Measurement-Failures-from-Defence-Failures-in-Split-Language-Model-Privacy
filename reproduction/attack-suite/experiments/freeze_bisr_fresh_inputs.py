"""Freeze 256/112 fresh article groups from pinned WikiText-103 training data.

Preserves previous freezes; verifies their exclusion-source hashes before use.
No models are trained and no attack outcomes are produced.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.prepare_pilot import articles
from experiments.freeze_bisr_inputs import sha, normal, text_hash, jsonl

REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def choose(candidates, count, salt, encode, forbidden_hashes, forbidden_titles,
           forbidden_windows, all_windows=False):
    """Reject first, then take deterministic hash order; every article contributes."""
    selected = []
    rejected = Counter()
    seen_hashes = set(forbidden_hashes)
    seen_titles = set(forbidden_titles)
    seen_windows = set(forbidden_windows)
    ordered = sorted(candidates, key=lambda d: (hashlib.sha256(
        (salt + normal(d['text'])).encode()).hexdigest(), d['id']))
    for doc in ordered:
        title = normal(doc['title']).casefold()
        if doc['normalised_sha256'] in seen_hashes or title in seen_titles:
            rejected['excluded_document_or_title'] += 1
            continue
        ids = encode(doc['text'])
        if len(ids) < 32:
            rejected['no_complete_window'] += 1
            continue
        windows = [tuple(ids[i:i+32]) for i in range(0, len(ids)-31, 32)] if all_windows else [tuple(ids[:32])]
        if any(w in seen_windows for w in windows):
            rejected['duplicate_or_forbidden_window'] += 1
            continue
        selected.append({**doc, 'token_ids': ids})
        seen_hashes.add(doc['normalised_sha256'])
        seen_titles.add(title)
        seen_windows.update(windows)
        if len(selected) == count:
            return selected, dict(rejected)
    raise ValueError(f'Only {len(selected)} of {count} admissible articles; no reduced-count fallback')


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir','previous-root','previous-manifest','tokenizer','amendment','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    previous = json.loads(args.previous_manifest.read_text())
    pins = {}
    paths = {}
    for name, pin in previous['source_pins'].items():
        path = Path(pin['path'])
        if not path.is_absolute(): path = args.previous_root/path
        if sha(path) != pin['sha256']: raise ValueError('Historical source changed: '+name)
        pins[name] = {'path':str(path), 'sha256':pin['sha256']}
        paths[name] = path
    pins['previous_manifest'] = {'path':str(args.previous_manifest),'sha256':sha(args.previous_manifest)}
    pins['current_amendment'] = {'path':str(args.amendment),'sha256':sha(args.amendment)}
    historical_hashes, historical_titles = set(), set()
    historical_texts = []

    def exclude(text, title=None):
        historical_hashes.add(text_hash(text))
        historical_texts.append(text)
        if title: historical_titles.add(normal(title).casefold())

    wiki2_ids = set()
    for split in ('train','validation','test'):
        path = paths['train'].parent/f'wikitext-{split}.arrow'
        pins['wiki2_'+split]={'path':str(path),'sha256':sha(path)}
        for title,text in articles(pa.ipc.open_stream(path).read_all()['text'].to_pylist()):
            exclude(text,title)
            wiki2_ids.add(f'wikitext2-{split}-'+hashlib.sha256(text.encode()).hexdigest()[:20])
    v3=json.loads(paths['v3'].read_text())
    if not {i for group in v3['selected_ids'].values() for i in group}.issubset(wiki2_ids):
        raise ValueError('Historical v3 identities do not resolve')
    for name in ('pilot','old_aux','b1_aux'):
        for doc in jsonl(paths[name]): exclude(doc['text'],doc.get('title'))
    for title,text in articles(paths['b1_local_corpus'].read_text().splitlines()): exclude(text,title)
    with paths['sensi'].open() as f:
        for row in csv.DictReader(f):
            for name in ('content','sani_gpt4'): exclude(row[name])
    # Explicitly verify and exclude both groups from the superseded freeze.
    for name in ('validation-documents.json','auxiliary-documents.json'):
        path=args.previous_manifest.parent/name
        if sha(path)!=previous['outputs'][name]: raise ValueError('Prior selected documents changed')
        pins['previous_'+name]={'path':str(path),'sha256':sha(path)}
        for doc in json.loads(path.read_text()): exclude(doc['text'],doc.get('title'))
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    tokenizer.model_max_length = 10**9  # tokenize complete articles; model only receives 32-token windows
    tokenizer_pins={name:sha(args.tokenizer/name) for name in previous['tokenizer_files']}
    if tokenizer_pins != previous['tokenizer_files']: raise ValueError('Tokenizer changed')
    # Prefix equality is an extra safeguard, not a near-duplicate certificate.
    historical_windows=set()
    for text in set(historical_texts):
        tokens=tokenizer.encode(text,add_special_tokens=False,truncation=True,max_length=32)
        if len(tokens)==32: historical_windows.add(tuple(tokens))
    shards=[args.source_dir/f'train-{i:05d}-of-00002.parquet' for i in range(2)]
    for i,path in enumerate(shards): pins[f'wiki103_train_{i}']={'path':str(path),'sha256':sha(path)}
    def rows():
        # Continuous stream preserves article boundaries across parquet shards.
        for path in shards:
            for batch in pq.ParquetFile(path).iter_batches(columns=['text'],batch_size=8192):
                yield from batch.column(0).to_pylist()
    candidates=[{'id':'wikitext103-train-'+hashlib.sha256(text.encode()).hexdigest()[:20],
                 'title':title,'text':text,'normalised_sha256':text_hash(text),'source_article_index':i}
                for i,(title,text) in enumerate(articles(rows()))]
    encode=lambda text:tokenizer.encode(text,add_special_tokens=False)
    validation, vr = choose(candidates,256,'bisr-conformance-v1',encode,
                             historical_hashes,historical_titles,historical_windows)
    val_hashes={d['normalised_sha256'] for d in validation}
    val_titles={normal(d['title']).casefold() for d in validation}
    val_windows={tuple(d['token_ids'][:32]) for d in validation}
    auxiliary, ar = choose(candidates,112,'bisr-aux-v1',encode,
        historical_hashes|val_hashes,historical_titles|val_titles,historical_windows|val_windows,True)
    validation=[{**d,'token_ids':d['token_ids'][:32],'mask':[1]*32} for d in validation]
    windows=[];counts={}
    for d in auxiliary:
        ids=d.pop('token_ids');counts[d['id']]=len(ids)//32
        for offset in range(0,len(ids)-31,32):
            windows.append({'sample_id':d['id']+f':{offset}','article_id':d['id'],
                            'token_ids':ids[offset:offset+32],'mask':[1]*32})
    args.output.mkdir(parents=True,exist_ok=False)
    files={'validation-documents.json':validation,'auxiliary-documents.json':auxiliary,
           'auxiliary-windows.json':windows,'truth.json':{'schema':'bisr.truth.v1',
            'sample_ids':[d['id'] for d in validation],'token_ids':[d['token_ids'] for d in validation]}}
    for name,value in files.items():
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    manifest={'schema':'bisr.frozen_inputs.v3','status':'INPUTS_FROZEN_CHECKPOINTS_NOT_ADMITTED',
      'dataset':'Salesforce/wikitext','configuration':'wikitext-103-raw-v1','revision':REVISION,
      'source_partition':'train','validation_design':'internal article-level holdout; not official test',
      'script_sha256':sha(__file__),'source_pins':pins,'tokenizer_files':tokenizer_pins,
      'validation_documents':256,'validation_ids':[d['id'] for d in validation],
      'auxiliary_articles':112,'auxiliary_windows':len(windows),'auxiliary_windows_by_article':counts,
      'reused_historical_auxiliary_articles':0,'zero_window_auxiliary_articles':0,
      'freshness_scope':'No exact normalised text/title overlap with pinned historical exclusions; not pretraining freshness',
      'historical_unique_document_hashes':len(historical_hashes),'historical_prefixes':len(historical_windows),
      'source_articles':len(candidates),'selection_rejections':{'validation':vr,'auxiliary':ar},
      'seeds':[31415,27182,16180],'equivalence_margin':0.01,
      'outputs':{name:sha(args.output/name) for name in files},'launch_admitted':False}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('validation_documents','auxiliary_articles','auxiliary_windows','reused_historical_auxiliary_articles','zero_window_auxiliary_articles')}))


if __name__=='__main__': main()
