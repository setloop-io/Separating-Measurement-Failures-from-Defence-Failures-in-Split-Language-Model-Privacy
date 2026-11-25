"""Tokenize explicitly partitioned documents and bind pre-outcome span annotations.

Input: JSON with calibration/train/evaluation arrays of {id,source_document_id,text,
optional char_annotations:[{kind,start,end}]} and provenance object. This command
never chooses partitions or reads reserved corpora implicitly.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attacker.evidence import file_sha256
from attacker.reconstruction_metrics import automatic_span_annotations
from experiments.validate_reconstruction import validate_data


def attach_annotations(raw,report):
    """Bind complete pre-outcome name review to source identity; no silent drops."""
    import hashlib
    rows=report['documents']
    expected={row['id']:row for row in raw['evaluation']}
    if len(rows)!=len(expected) or {r['id'] for r in rows}!=set(expected):
        raise ValueError('name annotation coverage/identity mismatch')
    if report['reviewed_document_count']!=len(expected) or not report.get('annotation_source'):
        raise ValueError('annotation provenance missing')
    for row in rows:
        source=expected[row['id']]
        if row.get('source_document_id',source['source_document_id'])!=source['source_document_id']:
            raise ValueError('annotation source identity mismatch')
        annotations=row['char_annotations']
        if any(a.get('kind')!='name' for a in annotations): raise ValueError('name review contains other categories')
        if 'prefix_text_sha256' in row:
            length=row['prefix_character_count']
            if hashlib.sha256(source['text'][:length].encode()).hexdigest()!=row['prefix_text_sha256']:
                raise ValueError('annotation text hash mismatch')
        source['char_annotations']=annotations


def prepare(source, output, protocol_path, annotations_path=None, tokenizer_dir=None):
    from transformers import AutoTokenizer
    if output.exists(): raise ValueError('output must be fresh')
    p=json.loads(protocol_path.read_text()); raw=json.loads(source.read_text())
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_dir or p['model'],revision=p['revision'],use_fast=True)
    annotation_report=json.loads(annotations_path.read_text()) if annotations_path else None
    if annotation_report is not None:
        attach_annotations(raw,annotation_report)
    data={'provenance':dict(raw['provenance'],source_sha256=file_sha256(source),model=p['model'],revision=p['revision'])}
    if annotations_path:
        data['provenance']['name_annotation_sha256']=file_sha256(annotations_path)
        data['provenance']['name_annotation_source']=annotation_report['annotation_source']
    source_ids=set()
    for partition in ('calibration','train','evaluation'):
        data[partition]=[]
        for row in raw[partition]:
            source_id=row['source_document_id']
            if not isinstance(source_id,str) or not source_id or source_id in source_ids:
                raise ValueError('each source document must occur once across all partitions')
            source_ids.add(source_id)
            encoded=tokenizer(row['text'],add_special_tokens=False,return_offsets_mapping=True,truncation=True,max_length=p['sequence_length'])
            offsets=encoded['offset_mapping'][:p['sequence_length']]
            tokens=encoded['input_ids'][:p['sequence_length']]
            annotations=automatic_span_annotations(row['text'],offsets)
            for annotation in row.get('char_annotations',[]):
                start,end=annotation['start'],annotation['end']
                if type(start) is not int or type(end) is not int or not 0<=start<end<=len(row['text']):
                    raise ValueError('invalid character annotation')
                covered=[i for i,(a,b) in enumerate(offsets) if a<end and b>start]
                if not covered or offsets[covered[0]][0]>start or offsets[covered[-1]][1]<end:
                    raise ValueError('annotation is outside or truncated by token window')
                item={'kind':annotation['kind'],'start':covered[0],'end':covered[-1]+1}
                if item not in annotations: annotations.append(item)
            data[partition].append({'id':row['id'],'source_document_id':source_id,'tokens':tokens,'annotations':annotations})
    validate_data(data,p,len(tokenizer))
    output.write_text(json.dumps(data,indent=2))
    print(json.dumps({'input':str(output),'sha256':file_sha256(output),'counts':{x:len(data[x]) for x in ('calibration','train','evaluation')}}))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--source',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,default=Path('experiments/validation_protocol.json'))
    ap.add_argument('--annotations',type=Path); ap.add_argument('--tokenizer-dir',type=Path)
    args=ap.parse_args(); prepare(args.source,args.output,args.protocol,args.annotations,args.tokenizer_dir)
