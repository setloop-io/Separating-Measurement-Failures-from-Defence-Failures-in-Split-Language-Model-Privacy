import json, math, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
out={}
rows=[]
for split in [21,19,17,14]:
 d=[]
 for s in [42,43,44]:
  p=ROOT/'outputs/phase07_noise_sweep'
  d.append(json.loads((p/f'n0.20_split{split}_s{s}.json').read_text())['eval_loss_delta']-json.loads((p/f'n0.35_split{split}_s{s}.json').read_text())['eval_loss_delta'])
 m=statistics.mean(d); half=4.302652729911275*statistics.stdev(d)/math.sqrt(3)
 rows.append(dict(split=split,paired_low_minus_high=d,mean=m,exploratory_t95=[m-half,m+half]))
out['noise_contrasts']=rows
p=ROOT/'outputs/phase09_economics_quality'
a=json.loads((p/'quality_intact.json').read_text())['results']; b=json.loads((p/'quality_composed.json').read_text())['results']
out['offline_quality']={'delta_nll':b['utility']['mean_nll']-a['utility']['mean_nll'],'answer_token_accuracy_drop_pp':100*(b['answer_top1']['token_match_rate']-a['answer_top1']['token_match_rate']),'note':'offline composed path bypasses wire defenses; teacher-forced answer tokens, not standard HellaSwag accuracy'}
(ROOT/'outputs/adversarial_review_2026-09-08/recomputed.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
