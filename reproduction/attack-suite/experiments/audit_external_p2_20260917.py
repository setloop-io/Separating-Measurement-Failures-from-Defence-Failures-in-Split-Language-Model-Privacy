#!/usr/bin/env python3
"""Historical raw-pair arithmetic audit only; does not admit retired evidence."""
import hashlib,json,math,statistics,itertools
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]/'results/external-p0p5-recovery-20260917'
ROOT=BASE/'toula-r1-analysis-bc5242e'

def t_sf(t,df):
    if t<0: return 1-t_sf(-t,df)
    a=math.atan(t/math.sqrt(df));b=math.pi/2;n=4096;h=(b-a)/n
    f=lambda x:math.cos(x)**(df-1)
    integral=h/3*(f(a)+f(b)+4*sum(f(a+i*h) for i in range(1,n,2))+2*sum(f(a+i*h) for i in range(2,n,2)))
    return math.exp(math.lgamma((df+1)/2)-.5*math.log(math.pi)-math.lgamma(df/2))*integral

def main():
    paired=[]
    for seed in [11,22,33,44,55,66,77,88]:
        d={}
        for mode in ['trained','frozen_random']:
            p=ROOT/f'outputs/confirmatory/{mode}-medium-high-s{seed}/metrics_summary.json'
            d[mode]=json.loads(p.read_text())
        assert d['trained']['pairing_block_id']==d['frozen_random']['pairing_block_id']
        assert d['trained']['remote_param_sha256_initial']==d['frozen_random']['remote_param_sha256_initial']
        assert d['frozen_random']['remote_param_sha256_initial']==d['frozen_random']['remote_param_sha256_final']
        steps={k:{c['step']:c['eval_loss'] for c in v['eval_series']} for k,v in d.items()}
        assert steps['trained'][0]==steps['frozen_random'][0]
        paired.append({'seed':seed,'trained_loss':d['trained']['final_eval_loss'],'frozen_random_loss':d['frozen_random']['final_eval_loss'],
                       'gain':d['frozen_random']['final_eval_loss']-d['trained']['final_eval_loss'],
                       'persistence_gains':{str(s):steps['frozen_random'][s]-steps['trained'][s] for s in [1800,1900,2000]}})
    g=[p['gain'] for p in paired];mu=statistics.mean(g);sd=statistics.stdev(g);se=sd/math.sqrt(len(g))
    analysis=json.loads((ROOT/'outputs/confirmatory/confirmatory_analysis.json').read_text())
    assert all(abs(p['gain']-analysis['cells']['medium-high']['final_gains'][str(p['seed'])])<1e-14 for p in paired)
    contrasts={}
    for margin in [0,.01]:
        x=[v-margin for v in g];order=sorted(range(len(x)),key=lambda i:abs(x[i]));ranks={i:r+1 for r,i in enumerate(order)}
        w=sum(ranks[i] for i in range(len(x)) if x[i]>0)
        sums=[sum(ranks[i] for i in range(len(x)) if bits[i]) for bits in itertools.product([0,1],repeat=len(x))]
        t=(mu-margin)/se
        contrasts[str(margin)]={'t_df7':t,'t_one_sided_p':t_sf(t,7),'wilcoxon_w_plus':w,'wilcoxon_exact_one_sided_p':sum(s>=w for s in sums)/len(sums),'positive_differences':sum(v>0 for v in x)}
    lo,hi=0.,10.
    for _ in range(60):
        mid=(lo+hi)/2
        if t_sf(mid,7)>.025:lo=mid
        else:hi=mid
    verified=[];missing=[];bad=[]
    for m in sorted((ROOT/'experiments/manifests').glob('*.json')):
        v=json.loads(m.read_text())
        for a in v.get('artifacts',[]):
            p=ROOT/a['path']
            if not p.is_file():missing.append({'manifest':m.name,'path':a['path']});continue
            actual=hashlib.sha256(p.read_bytes()).hexdigest()
            (verified if actual==a['sha256'] else bad).append({'manifest':m.name,'path':a['path'],'sha256':actual})
    result={'classification':'historical arithmetic audit; pre-reset data inadmissible for active claims',
            'paired_losses':paired,'mean':mu,'sample_sd':sd,'se':se,'two_sided_95_ci':[mu-hi*se,mu+hi*se],'contrasts':contrasts,
            'matches_historical_analysis':True,'recovered_manifest_artifact_checks':{'verified_references':len(verified),'mismatched':bad,'not_downloaded_references':len(missing)},
            'method':'Student t tail via Simpson integration of transformed cos^(df-1) density (4096 intervals); exact Wilcoxon via all 2^8 sign assignments; no ties/zeros in these pairs'}
    (BASE/'p2-paired-statistics-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (BASE/'manifest-artifact-verification.json').write_text(json.dumps({'verified':verified,'mismatched':bad,'not_downloaded':missing},indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['paired_losses']},indent=2))
if __name__=='__main__':main()
