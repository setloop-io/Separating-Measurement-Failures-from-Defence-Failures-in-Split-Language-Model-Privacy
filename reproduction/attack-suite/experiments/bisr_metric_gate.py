"""Check pinned upstream metric parity on fixed strings, never heldout outcomes."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from attacker.bisr_fidelity import score_predictions,sha256
from experiments.bisr_upstream_preflight import extract,source,save


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('tokenizer','upstream','wordnet','output'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    from transformers import AutoTokenizer
    from rouge import Rouge
    from nltk.translate import meteor_score
    tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    src=source(args.upstream,'sfl/utils/model.py')
    env={'Rouge':Rouge,'meteor_score':meteor_score}
    rg=extract(src,'calculate_rouge_text',env)
    mt=extract(src,'calculate_meteor',env)
    tf=extract(src,'calculate_token_acc',env)
    pairs=[('The quick brown fox.','The slow brown dog.'),
           ('The car stopped.','The automobile stopped.'),
           ('A cat sat here.','A cat sat here.'),
           ('','A cat sat here.')]
    rows=[]
    for i,(hyp,ref) in enumerate(pairs):
        pred=tok.encode(hyp,add_special_tokens=False);truth=tok.encode(ref,add_special_tokens=False)
        n=max(len(pred),len(truth));pred += [tok.eos_token_id]*(n-len(pred));truth += [tok.eos_token_id]*(n-len(truth))
        record={'schema':'bisr.predictions.v1','sample_ids':[str(i)],
                'arms':{a:[pred] for a in ('sip','b','f','bf')}}
        score=score_predictions(record,{'schema':'bisr.truth.v1','sample_ids':[str(i)],'token_ids':[truth]},tok)['sip'][0]
        h=' '.join(tok.decode(t,skip_special_tokens=True) for t in pred)
        r=' '.join(tok.decode(t,skip_special_tokens=True) for t in truth)
        # The local scorer defines whitespace-only hypotheses as empty. Use that
        # declared canonicalisation on both sides before calling upstream metrics.
        h=h if h.strip() else '<EMP>'
        reference={'positional_accuracy':sum(a==b for a,b in zip(pred,truth))/len(truth),
                   'rouge_l_f1':rg([h],[r])['rouge-l']['f'],
                   'meteor':mt([h],[r]),'token_set_f1':tf(tok,[h],[r])}
        differences={k:abs(score[k]-v) for k,v in reference.items()}
        if any(d>1e-12 for d in differences.values()):raise ValueError('Metric mismatch')
        rows.append({'hypothesis':h,'reference':r,'candidate':score,'upstream':reference,'absolute_differences':differences})
    save(args.output,{'status':'METRIC_PARITY_PASS','scope':'fixed strings with declared empty-hypothesis canonicalisation',
        'cases':rows,'tolerance':1e-12,'upstream_source_sha256':hashlib.sha256(src.encode()).hexdigest(),
        'wordnet_sha256':sha256(args.wordnet),'script_sha256':sha256(__file__),
        'versions':{p:importlib.metadata.version(p) for p in ('rouge','nltk','transformers')},
        'fidelity_established':False})


if __name__=='__main__':main()
