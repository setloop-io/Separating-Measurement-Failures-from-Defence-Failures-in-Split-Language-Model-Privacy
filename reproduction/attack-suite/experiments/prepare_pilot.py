"""Build the preregistered article-level pilot sample from pinned WikiText parquet."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re


def articles(rows):
    title, paragraphs = None, []
    for text in rows:
        text = text.strip()
        if re.fullmatch(r'= [^=].*[^=] =', text):
            if title and paragraphs:
                yield title, '\n\n'.join(paragraphs)
            title, paragraphs = text[2:-2].strip(), []
        elif title and text and not text.startswith('='):
            paragraphs.append(text)
    if title and paragraphs:
        yield title, '\n\n'.join(paragraphs)


def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', required=True)
    args=ap.parse_args()
    root=Path(args.inputs)
    tokenizer=AutoTokenizer.from_pretrained(root/'tokenizer')
    candidates=[]; seen=set(); windows=set(); grams=[]; dropped=0
    for title, text in articles(pq.read_table(root/'train.parquet')['text'].to_pylist()):
        digest=hashlib.sha256(text.encode()).hexdigest()
        tokens=tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=32)
        if len(tokens)<32 or digest in seen or tuple(tokens) in windows:
            dropped+=1;continue
        # Conservative duplicate filtering on the scored token prefix.
        g={tuple(tokens[i:i+5]) for i in range(len(tokens)-4)}
        if any(len(g & prior)/len(g | prior)>=.8 for prior in grams):
            dropped+=1;continue
        seen.add(digest); windows.add(tuple(tokens)); grams.append(g)
        candidates.append({'id': 'wikitext2-train-'+digest[:20], 'text':text,'title':title})
    random.Random(20260908).shuffle(candidates)
    chosen=candidates[:96]
    if len(chosen)!=96: raise ValueError('fewer than 96 eligible articles')
    output=root/'pilot-corpus.jsonl'
    output.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in chosen))
    manifest={'schema':'llm_attacker.pilot_corpus.v1','dataset':'Salesforce/wikitext',
              'revision':'b08601e04326c79dfdd32d625aee71d232d685c3','configuration':'wikitext-2-raw-v1',
              'source_split':'train','reserved_splits':['validation','test'],
              'source_url':'https://huggingface.co/datasets/Salesforce/wikitext',
              'model_revision':'607a30d783dfa663caf39e06633721c8d4cfcd7e',
              'selection_seed':20260908,'partition_seed':2026,'sequence_length':32,
              'selected_articles':len(chosen),'eligible_articles':len(candidates),'excluded_articles':dropped,
              'deduplication':'exact full article text and token prefix; first-32-token 5-gram Jaccard >=0.8 excluded',
              'limits':['public benchmark documents stand in for private inputs','near-duplicate filtering covers scored prefixes only',
                        'not a membership study; pretrained corpus overlap cannot be independently certified'],
              'files':{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'train.parquet',output]},
              'selected_ids':[d['id'] for d in chosen]}
    (root/'corpus-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('selected_ids','files')},indent=2))


if __name__=='__main__': main()
