"""Deterministic article-disjoint follow-up sample; excludes the entire pilot.

Only the pinned official training split is opened. Official validation/test remain
reserved. This creates annotation inputs, not attack outcomes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attacker.evidence import file_sha256
from experiments.prepare_pilot import articles
from attacker.reconstruction_metrics import automatic_span_annotations

SOURCE_SHA='e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7'
PILOT_SHA='d11d88a6c0331024683e462814aff07d8158ae3829487d714bf77086bfbb7c77'
SELECTION_SALT='validation-v1-20260908-article-disjoint'


def grams(tokens): return {tuple(tokens[i:i+5]) for i in range(len(tokens)-4)}
def title_key(title): return re.sub(r'\s+',' ',title).strip().casefold()


def curate(inputs,output):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    if output.exists(): raise ValueError('fresh output required')
    if file_sha256(inputs/'train.parquet')!=SOURCE_SHA or file_sha256(inputs/'pilot-corpus.jsonl')!=PILOT_SHA:
        raise ValueError('pinned source or pilot exclusion hash mismatch')
    tokenizer=AutoTokenizer.from_pretrained(inputs/'tokenizer',use_fast=True)
    old=[json.loads(line) for line in (inputs/'pilot-corpus.jsonl').read_text().splitlines()]
    excluded_text={hashlib.sha256(r['text'].encode()).hexdigest() for r in old}
    excluded_titles={title_key(r['title']) for r in old}
    prior_grams=[grams(tokenizer.encode(r['text'],add_special_tokens=False,truncation=True,max_length=32)) for r in old]
    candidates=[]; discarded=[]
    for title,text in articles(pq.read_table(inputs/'train.parquet')['text'].to_pylist()):
        digest=hashlib.sha256(text.encode()).hexdigest()
        encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,truncation=True,max_length=32)
        tokens=encoded['input_ids']; token_grams=grams(tokens)
        reason=None
        if digest in excluded_text or title_key(title) in excluded_titles: reason='pilot article identity'
        elif len(tokens)!=32: reason='short prefix'
        elif any(len(token_grams & g)/len(token_grams | g)>=.8 for g in prior_grams): reason='near duplicate of pilot prefix'
        if reason:
            discarded.append({'text_sha256':digest,'reason':reason});continue
        candidates.append({'id':'wikitext2-train-'+digest[:20], 'source_document_id':digest,
                           'title':title,'text':text,'tokens':tokens,'offsets':encoded['offset_mapping']})
    candidates.sort(key=lambda r:hashlib.sha256((SELECTION_SALT+r['source_document_id']).encode()).hexdigest())
    selected=[]; seen_titles=set(); seen_text=set(); selected_grams=[]
    for row in candidates:
        g=grams(row['tokens']); digest=row['source_document_id']
        if digest in seen_text or title_key(row['title']) in seen_titles or any(len(g & prior)/len(g | prior)>=.8 for prior in selected_grams):
            discarded.append({'text_sha256':digest,'reason':'duplicate among new articles'});continue
        selected.append(row); seen_text.add(digest);seen_titles.add(title_key(row['title']));selected_grams.append(g)
    if len(selected)<416: raise ValueError(f'only {len(selected)} independent candidates, need416')
    output.mkdir(parents=True)
    raw={'provenance':{'untouched_evaluation':True,'meaning':'No selected evaluation article was used in the earlier pilot or baseline attacks; annotation/curation sees text before outcomes.',
         'source_split':'official train; new article-level internal holdout','dataset':'Salesforce/wikitext','configuration':'wikitext-2-raw-v1',
         'revision':'b08601e04326c79dfdd32d625aee71d232d685c3','source_sha256':SOURCE_SHA,'excluded_pilot_sha256':PILOT_SHA,
         'selection_salt':SELECTION_SALT,'reserved_official_splits':['validation','test'],
         'limitations':['Original curation scanned training articles; untouched means outcome-unseen, not never parsed.',
                        'Pretraining exposure cannot be excluded; this is a reconstruction study, not membership inference.',
                        'Only scored prefixes receive near-duplicate filtering; broader semantic similarity may remain.']}}
    boundaries=[('calibration',0,32),('train',32,160),('evaluation',160,416)]
    annotation_rows=[]
    for name,start,end in boundaries:
        raw[name]=[]
        for row in selected[start:end]:
            raw[name].append({k:row[k] for k in ('id','source_document_id','title','text')})
            if name=='evaluation':
                end_char=max(b for a,b in row['offsets'])
                annotation_rows.append({'id':row['id'],'title':row['title'],'prefix_text':row['text'][:end_char],
                                        'token_offsets':row['offsets'], 'automatic_token_annotations':automatic_span_annotations(row['text'],row['offsets'])})
    (output/'partitioned-documents.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2))
    (output/'annotation-input.json').write_text(json.dumps(annotation_rows,ensure_ascii=False,indent=2))
    report={'source_sha256':SOURCE_SHA,'excluded_pilot_sha256':PILOT_SHA,'selection_salt':SELECTION_SALT,
            'eligible_after_exclusions':len(selected),'partition_counts':{key:len(raw[key]) for key,_,_ in boundaries},
            'selected_ids':{key:[r['id'] for r in raw[key]] for key,_,_ in boundaries},'discarded':discarded,
            'outputs':{p.name:file_sha256(p) for p in output.iterdir()}}
    (output/'curation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({key:report[key] for key in ('eligible_after_exclusions','partition_counts')},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();curate(args.inputs,args.output)
