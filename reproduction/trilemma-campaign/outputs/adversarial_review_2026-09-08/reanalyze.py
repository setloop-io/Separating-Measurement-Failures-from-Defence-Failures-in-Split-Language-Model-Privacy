"""Post-hoc seed-level contrasts; no training or original-artifact edits."""
import json, sys, hashlib, itertools, math
from pathlib import Path
from statistics import mean, stdev
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from stats_utils import paired_t_test
OUT=Path(__file__).parent
sources={}
def loss(group,rid):
 p=ROOT/'outputs'/group/rid/'metrics_summary.json'
 b=p.read_bytes(); sources[str(p.relative_to(ROOT))]=hashlib.sha256(b).hexdigest()
 return json.loads(b)['final_eval_loss']
def contrast(name,group,a,b,seeds):
 vals=[loss(group,a(s))-loss(group,b(s)) for s in seeds]
 stat=paired_t_test(vals)
 observed=abs(sum(vals)); exceed=sum(abs(sum(s*x for s,x in zip(signs,vals)))>=observed-1e-12 for signs in itertools.product((-1,1),repeat=len(vals)))
 return dict(name=name,seeds=seeds,values=vals,**stat,sign_flip_p_two_sided=exceed/(2**len(vals)))
r=[]
r.append(contrast('P2 medium high: removed minus trained','confirmatory',lambda s:f'removed_remote-medium-s{s}',lambda s:f'trained-medium-high-s{s}',[11,22,33,44,55,66,77,88]))
for q in (4,8,16,32):
 r.append(contrast(f'P2b q{q}: removed minus trained','rate_sweep',lambda s:f'removed_remote-medium-q{q}-s{s}',lambda s:f'trained-medium-high-q{q}-s{s}',[7,17,27,37,47,57,67,87]))
for setting in ('a2c1','a1c2','a2c2'):
 r.append(contrast(f'P3 {setting}: removed minus trained','replication',lambda s:f'{setting}-removed_remote-s{s}',lambda s:f'{setting}-trained-s{s}',[9,19,29,39,49,59,69,89]))
for span,start in ((1,3),(2,3),(3,2),(4,2),(5,1),(6,1),(7,0)):
 r.append(contrast(f'P5 s{span} d0: removed minus trained','delegation',lambda s:f'p5-removed-s{span}at{start}-s{s}',lambda s:f'p5-s{span}-d0-trained-s{s}',[21,31,41,51,61,71,81,91]))
for span,early,late in ((2,1,5),(3,2,4),(5,1,3)):
 r.append(contrast(f'P2c span{span}: early minus late loss','segment_stats',lambda s:f'trained-s{span}at{early}-high-s{s}',lambda s:f'trained-s{span}at{late}-high-s{s}',[3,13,23,43,53,63,83]))
r.append(contrast('P2b q8 minus q32 trained loss','rate_sweep',lambda s:f'trained-medium-high-q8-s{s}',lambda s:f'trained-medium-high-q32-s{s}',[7,17,27,37,47,57,67,87]))
# Holm over ALL listed exploratory t-tests, including weak and negative results.
order=sorted(range(len(r)),key=lambda i:r[i]['p']); running=0
for k,i in enumerate(order):
 running=max(running,min(1,r[i]['p']*(len(r)-k)));r[i]['holm_p_this_19_contrast_family']=running
(OUT/'contrasts.json').write_text(json.dumps({'status':'post-hoc exploratory; not newly preregistered confirmation','positive_removed_minus_trained':'trained outperforms identity bypass','results':r,'source_sha256':sources},indent=2))
for x in r: print(x['name'],f"mean={x['mean']:.6f} CI=[{x['ci95_low']:.6f},{x['ci95_high']:.6f}] p={x['p']:.6g} Holm={x['holm_p_this_19_contrast_family']:.6g} signflip={x['sign_flip_p_two_sided']:.6g}")
# Verify every referenced metrics file against its current manifest entry.
checks=[]
for rel,h in sources.items():
 rid=Path(rel).parent.name;p=ROOT/'experiments/manifests'/f'{rid}.json'
 m=json.loads(p.read_text());entry=next((a for a in m['artifacts'] if a['path']==rel),None)
 checks.append({'path':rel,'manifest_match':bool(entry and entry['sha256']==h)})
(OUT/'hash_checks.json').write_text(json.dumps(checks,indent=2))
print('Metric hashes matched:',sum(x['manifest_match'] for x in checks),'/',len(checks))
