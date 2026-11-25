"""Validate the complete pinned pilot and regenerate its tables and findings."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacker.experiment import interval, verify_run
from attacker.evidence import EvidenceLogger, file_sha256


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def plot_recovery(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),sharey=True,layout='constrained')
    for row,mode in enumerate(['training','inference']):
        for col,depth in enumerate([3,6]):
            ax=axes[row,col];arms=['vmi','wbi','joint'] if mode=='training' else ['vmi','wbi']
            for offset,defense,color in [(-.18,'none','#476582'),(.18,'gaussian','#ca7440')]:
                data=[next(r for r in summary if (r['depth'],r['mode'],r['defense'],r['arm'])==(depth,mode,defense,a)) for a in arms]
                ax.bar([i+offset for i in range(len(arms))],[r['accuracy_pct_mean'] for r in data],
                       yerr=[r['accuracy_pct_sd'] for r in data],width=.34,color=color,capsize=3,
                       label='Undefended' if defense=='none' else 'Gaussian σ=0.1, adaptive attacker')
            ax.axhline(data[0]['control_pct_mean'],color='#666666',linestyle='--',linewidth=1,label='Majority control')
            ax.set_xticks(range(len(arms)),[a.upper() if a!='joint' else 'Joint' for a in arms]);ax.set_ylim(0,100)
            ax.set_title(f'{mode.capitalize()} · split after layer {depth}')
            ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            if col==0:ax.set_ylabel('Position-aligned token recovery (%)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=3,frameon=False)
    fig.suptitle('Pretrained GPT-2 pilot · mean ± SD across five seeds',fontsize=14)
    for extension in ['png','pdf','svg']:fig.savefig(out/f'recovery.{extension}',dpi=180)
    plt.close(fig)


def analyze(root, protocol_path):
    protocol=json.loads(protocol_path.read_text())
    records=[];anchors={};initial_hashes=set();partitions=None
    expected={(seed,defense,mode,arm) for seed in protocol['seeds'] for defense in protocol['defenses']
              for mode,arms in protocol['attack_arms'].items() for arm in arms}
    for depth in protocol['split_depths']:
        folder=root/f'split{depth}'
        anchor_path=root/f'origin-anchor-split{depth}.json'
        anchor=json.loads(anchor_path.read_text())['manifest_sha256'] if anchor_path.exists() else None
        ok,problems=verify_run(folder,anchor)
        if not ok:raise ValueError(f'split {depth} verification failed: {problems}')
        raw=json.loads((folder/'experiment.json').read_text())
        if raw['evidence_tier']!='local_model_experiment':raise ValueError('a smoke run is not pilot evidence')
        if raw['corpus_sha256']!=protocol['corpus_sha256']:raise ValueError('corpus differs from frozen protocol')
        config=raw['config']
        for field in ['seeds','seq_len','train_steps','decode_steps','attack_steps','calibration_seqs','joint_beam','joint_candidates','data_seed','adaptive','defenses']:
            if config[field]!=protocol[field]:raise ValueError(f'config drift: {field}')
        if config['sigma']!=protocol['gaussian_sigma'] or config['lr']!=protocol['training_lr'] or int(config['split'])!=depth:
            raise ValueError('defense/training/split configuration drift')
        keys=[(r['seed'],r['defense'],r['mode'],r['arm']) for r in raw['results']]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('missing, duplicate or unexpected cells')
        if partitions is not None and partitions!=raw['partitions']:raise ValueError('document partition drift')
        partitions=raw['partitions']
        manifest=json.loads((folder/'experiment.json.evidence.json').read_text())
        if manifest['git']['commit']!=protocol['source_commit'] or manifest['git']['dirty']:
            raise ValueError('engine is not the clean pinned release commit')
        anchors[str(depth)]={'manifest_sha256':manifest['manifest_sha256'],'packages':manifest['packages'],
                             'source_sha256':manifest['git']['source_sha256'],'resources':raw['resources'],
                             'wall_seconds':raw['wall_seconds']}
        for r in raw['results']:
            r={**r,'depth':depth};records.append(r);initial_hashes.add(r['public_model_sha256'])
            if r['n_sequences']!=32 or r['n_tokens']!=1024:raise ValueError('scoring count drift')
    if len(initial_hashes)!=1:raise ValueError('public initial weights differ across runs')
    if len(records)!=protocol['expected_cells']:raise ValueError('incomplete pilot')
    comparison_family=protocol['statistics']['family_size']
    baseline={(r['depth'],r['seed'],r['mode'],r['arm']):r for r in records if r['defense']=='none'}
    groups={}
    for r in records:groups.setdefault((r['depth'],r['mode'],r['defense'],r['arm']),[]).append(r)
    summary=[];seed_rows=[];sequence_rows=[];utility={}
    for (depth,mode,defense,arm),cells in sorted(groups.items()):
        cells.sort(key=lambda r:r['seed'])
        attack=[r['accuracy']*100 for r in cells]
        effect=[100*(r['accuracy']-r['control_accuracy']) for r in cells]
        paired=[100*(r['accuracy']-baseline[(depth,r['seed'],mode,arm)]['accuracy']) for r in cells]
        shuffled=[100*(r['accuracy']-r['shuffled_pair_accuracy']) for r in cells]
        bounds=interval(effect,comparison_family);paired_bounds=interval(paired,comparison_family);null_bounds=interval(shuffled,comparison_family)
        row={'depth':depth,'mode':mode,'defense':defense,'arm':arm,'seeds':len(cells),
             'accuracy_pct_mean':statistics.mean(attack),'accuracy_pct_sd':statistics.stdev(attack),
             'control_pct_mean':statistics.mean(r['control_accuracy']*100 for r in cells),
             'shuffled_pair_pct_mean':statistics.mean(r['shuffled_pair_accuracy']*100 for r in cells),
             'attack_minus_control_pp':bounds['mean'],'control_difference_ci_low':bounds['lower'],'control_difference_ci_high':bounds['upper'],
             'defense_minus_none_pp':paired_bounds['mean'],'defense_difference_ci_low':paired_bounds['lower'],'defense_difference_ci_high':paired_bounds['upper'],
             'attack_minus_shuffled_pp':null_bounds['mean'],'shuffled_difference_ci_low':null_bounds['lower'],'shuffled_difference_ci_high':null_bounds['upper'],
             'exact_sequence_match_pct':statistics.mean(r['exact_sequence_match']*100 for r in cells),
             'token_set_precision_pct':statistics.mean(r['attack_summary'][0]['token_set_precision_mean']*100 for r in cells),
             'token_set_recall_pct':statistics.mean(r['attack_summary'][0]['token_set_recall_mean']*100 for r in cells),
             'attack_seconds_mean':statistics.mean(r['attack_seconds'] for r in cells),
             'joint_minus_forward_pp':statistics.mean(r['attack_summary'][0]['joint_minus_fwd_pp_mean'] for r in cells) if arm=='joint' else None,
             'joint_minus_backward_pp':statistics.mean(r['attack_summary'][0]['joint_minus_bwd_pp_mean'] for r in cells) if arm=='joint' else None}
        summary.append(row)
        for r in cells:
            seed_rows.append({k:r[k] for k in ['depth','mode','defense','arm','seed','accuracy','control_accuracy','shuffled_pair_accuracy','exact_sequence_match','n_sequences','n_tokens','attack_seconds']})
            utility[(depth,defense,r['seed'])]={'depth':depth,'defense':defense,'seed':r['seed'],
                'ce_before':r['utility_before']['cross_entropy'],'ce_after':r['utility_after']['cross_entropy'],
                'ppl_before':r['utility_before']['perplexity'],'ppl_after':r['utility_after']['perplexity'],
                'training_seconds':r['training']['wall_seconds'],'inference_seconds':r['inference']['wall_seconds'],
                'client_parameter_change_l2':r['training']['client_parameter_change_l2']}
            for q in r['per_sequence']:
                sequence_rows.append({**{k:r[k] for k in ['depth','mode','defense','arm','seed']},
                                      **{k:q[k] for k in ['index','document_id','n_tokens','correct','control_correct','shuffled_pair_correct']}})
    out=root/'analysis';out.mkdir(exist_ok=True)
    write_csv(out/'summary.csv',summary);write_csv(out/'seed_metrics.csv',seed_rows)
    write_csv(out/'sequence_metrics.csv',sequence_rows);write_csv(out/'utility.csv',list(utility.values()))
    diagnostics=[]
    for depth in protocol['split_depths']:
        path=root/f'candidate-ceiling-split{depth}.json'
        if path.exists():
            diagnostic=json.loads(path.read_text())
            if diagnostic['source_experiment_sha256']!=file_sha256(root/f'split{depth}'/'experiment.json'):
                raise ValueError('diagnostic source mismatch')
            if diagnostic['script_sha256']!=file_sha256(Path(__file__).with_name('candidate_ceiling.py')):
                raise ValueError('diagnostic script mismatch')
            diagnostics.extend(diagnostic['results'])
    result={'schema' :'llm_attacker.pilot_analysis.v1','status':'COMPLETE','cells':len(records),
            'model_weight_sha256':next(iter(initial_hashes)),'protocol_sha256':file_sha256(protocol_path),
            'analysis_script_sha256':file_sha256(__file__),'anchors':anchors,'summary':summary,
            'utility':list(utility.values()),'comparison_family_size':comparison_family,
            'candidate_ceiling_diagnostic':diagnostics,
            'interpretation':'Pilot estimates conditional on the selected documents, five seeds, access and budgets. Difference intervals are Bonferroni-adjusted over both depths. Zero-crossing is inconclusive, not equivalence.',
            'limitations':protocol['limitations']}
    lines=['# Pretrained GPT-2 / WikiText-2 pilot','',
           f"Completed {len(records)}/{protocol['expected_cells']} cells: 2 split depths × 2 defenses × 5 seeds × 5 mode/attack combinations.",'',
           '96 articles: 32 calibration, 32 victim-training, 32 held-out evaluation. Each scored window has 32 tokens. The official validation/test splits were reserved.', '',
           'The engine is the clean V.0.1 release. All input and artifact hashes verified; all runs used identical initial public model weights.', '',
           '## Pilot findings','',
           '1. Activation-only VMI recovered roughly 80–89% of position-aligned tokens across the tested conditions, compared with majority controls of 3.42% (training) and 4.30% (inference).',
           '2. At Gaussian sigma=0.1, every paired defense-minus-undefended interval includes zero after the frozen multiplicity correction. This pilot does not demonstrate a recovery reduction at that setting; it does not establish equivalence or evaluate stronger noise.',
           '3. The joint decoder improves on its backward-only diagnostic but performs below forward-only VMI. The post-hoc candidate-set audit below identifies the limiting mechanism.',
           '4. Moving the split from layer 3 to layer 6 decreases VMI recovery while increasing the backward/joint decoder recovery in this sample. Treat this as a follow-up hypothesis: depth is assigned to host, and these are particular implementations.',
           '5. Held-out perplexity improves after training. The undefended runs produce matching utility at both depths; this is useful evidence that changing the split preserves the underlying training computation.', '',
           'The next methodological priority is a stronger joint candidate construction and an attack-budget comparison. This pilot does not support a general claim that joint observations are harmless or that the Gaussian family is ineffective.', '',
           '## Recovery and paired uncertainty','', 
           'Accuracy is mean ± sample SD across five seeds. Difference intervals use the frozen 60-comparison family across both depths; they are conditional on this corpus, not population guarantees. Negative defense-minus-none values mean reduced recovery.', '',
           '| Split | Mode | Defense | Arm | Accuracy % ± SD | Majority % | Attack − majority pp [CI] | Defense − none pp [CI] | Attack seconds |',
           '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['depth']} | {r['mode']} | {r['defense']} | {r['arm']} | {r['accuracy_pct_mean']:.2f} ± {r['accuracy_pct_sd']:.2f} | {r['control_pct_mean']:.2f} | {r['attack_minus_control_pp']:.2f} [{r['control_difference_ci_low']:.2f}, {r['control_difference_ci_high']:.2f}] | {r['defense_minus_none_pp']:.2f} [{r['defense_difference_ci_low']:.2f}, {r['defense_difference_ci_high']:.2f}] | {r['attack_seconds_mean']:.2f} |")
    lines+=['','## Held-out utility','', 'Each model condition is counted once per seed, despite appearing in several attack records. Perplexities below are arithmetic means of per-run perplexities.', '',
            '| Split | Defense | CE before → after | Perplexity before → after | Training seconds |','|---|---|---:|---:|---:|']
    for depth,defense in itertools.product(protocol['split_depths'],protocol['defenses']):
        cells=[r for r in utility.values() if r['depth']==depth and r['defense']==defense]
        means={k:statistics.mean(r[k] for r in cells) for k in ['ce_before','ce_after','ppl_before','ppl_after','training_seconds']}
        lines.append(f"| {depth} | {defense} | {means['ce_before']:.3f} → {means['ce_after']:.3f} | {means['ppl_before']:.2f} → {means['ppl_after']:.2f} | {means['training_seconds']:.2f} |")
    lines+=['','## Joint observation diagnostics','',
            'These are descriptive comparisons with the implemented single-surface decoders, not equal-compute comparisons to the published strongest attacks.', '',
            '| Split | Defense | Joint − forward pp | Joint − backward pp |','|---|---|---:|---:|']
    for r in summary:
        if r['arm']=='joint':lines.append(f"| {r['depth']} | {r['defense']} | {r['joint_minus_forward_pp']:.2f} | {r['joint_minus_backward_pp']:.2f} |")
    if diagnostics:
        lines+=['','## Post-hoc candidate-filter diagnostic','',
                'The joint decoder considers only backward top-4 candidates after position zero. The following oracle ceiling grants perfect position-zero recovery and perfect selection among all remaining candidates. It is computed from the original saved captures, without changing the frozen attacks or budgets.', '',
                '| Split | Defense | Candidate-set oracle ceiling % | Actual joint % | Forward-only % |',
                '|---|---|---:|---:|---:|']
        for depth,defense in itertools.product(protocol['split_depths'],protocol['defenses']):
            cells=[r for r in diagnostics if r['depth']==depth and r['defense']==defense]
            lines.append(f"| {depth} | {defense} | {100*statistics.mean(r['oracle_sequence_accuracy_ceiling'] for r in cells):.2f} | {100*statistics.mean(r['actual_joint_accuracy'] for r in cells):.2f} | {100*statistics.mean(r['forward_accuracy'] for r in cells):.2f} |")
        lines+=['','This identifies a limitation of candidate construction in this implementation. Increasing beam width alone cannot overcome a missing correct token at fixed top-k. It does not show that the combined observation contains less information or establish an optimal joint attack.','']
    lines+=['','## Scope and limitations','']+['- '+x for x in protocol['limitations']]
    lines+=['','## Primary model and data sources','',
            '- [GPT-2 model card](https://huggingface.co/openai-community/gpt2)',
            '- [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext)','',
            'Regenerate: `python experiments/analyze_pilot.py --root results/pilot-gpt2-wikitext2`','']
    (out/'report.md').write_text('\n'.join(lines))
    plot_recovery(summary,out)
    result['files']={p.name:file_sha256(p) for p in out.iterdir() if p.suffix in ['.csv','.md','.png','.pdf','.svg']}
    artifact=out/'analysis.json'
    with EvidenceLogger(artifact,extra={'protocol_sha256':result['protocol_sha256']}) as ev:
        ev.event('verified_pilot',{'cells':len(records),'anchors':{d:v['manifest_sha256'] for d,v in anchors.items()}})
        artifact.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'status':'COMPLETE','cells':len(records),'report':str(out/'report.md')},indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,default=Path('experiments/pilot_gpt2_wikitext2.json'))
    a=ap.parse_args();analyze(a.root,a.protocol)


if __name__=='__main__':main()
