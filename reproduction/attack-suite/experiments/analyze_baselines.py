"""Verify and analyze the complete frozen reconstruction comparison."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacker.evidence import EvidenceLogger, file_sha256, verify_evidence
from attacker.experiment import interval, verify_run
from experiments.compare_baselines import expected_arms


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def verify_comparison(folder, protocol, source_root):
    anchor = json.loads((folder.parent / f'origin-anchor-{folder.name}.json').read_text())['manifest_sha256']
    ok, problems = verify_evidence(folder / 'comparison.json.evidence.json', anchor)
    if not ok:
        raise ValueError(f'comparison evidence failed: {problems}')
    raw = json.loads((folder / 'comparison.json').read_text())
    depth = raw['depth']
    if folder.name != f'split{depth}':
        raise ValueError('depth/folder mismatch')
    manifest = json.loads((folder / 'comparison.json.evidence.json').read_text())
    if manifest['git']['commit'] != '6292d4aafa88c8f7fbe92853531c242cf2cf8c82' or manifest['git']['dirty']:
        raise ValueError('execution is not the clean frozen baseline commit')
    setup_keys = [r['id'] for r in raw['setup']]
    expected_setup = {f'{seed}-{defense}' for seed in protocol['seeds'] for defense in protocol['defenses']}
    if len(set(setup_keys)) != len(setup_keys) or set(setup_keys) != expected_setup:
        raise ValueError('missing or duplicate setup measurements')
    if raw['status'] != 'COMPLETE' or raw['protocol'] != protocol:
        raise ValueError('incomplete run or protocol drift')
    expected = {(seed, defense, mode, arm) for seed in protocol['seeds'] for defense in protocol['defenses']
                for mode in protocol['modes'] for arm in expected_arms(protocol, mode)}
    keys = [(r['seed'], r['defense'], r['mode'], r['arm']) for r in raw['results']]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('missing, duplicate or unexpected comparison cells')
    source_folder = source_root / f'split{depth}'
    ok, problems = verify_run(source_folder, protocol['source_manifest_sha256'][str(depth)])
    if not ok:
        raise ValueError(f'source pilot verification failed: {problems}')
    if raw['source_manifest_sha256'] != protocol['source_manifest_sha256'][str(depth)] or raw['source_experiment_sha256'] != file_sha256(source_folder / 'experiment.json'):
        raise ValueError('source anchor drift')
    source = json.loads((source_folder / 'experiment.json').read_text())
    truths = {(r['seed'], r['defense'], r['mode']): {q['document_id']: q for q in r['per_sequence']}
              for r in source['results'] if r['arm'] == 'vmi'}
    for cell in raw['results']:
        if cell['setup_id'] != f"{cell['seed']}-{cell['defense']}":
            raise ValueError('setup/cell mismatch')
        pred_path = (folder / cell['prediction_file']).resolve()
        if not pred_path.is_relative_to(folder.resolve()) or file_sha256(pred_path) != cell['prediction_sha256']:
            raise ValueError('prediction artifact changed')
        pred = json.loads(pred_path.read_text())
        if 'candidate_ids' in pred['diagnostics']:
            candidate_rows = pred['diagnostics']['candidate_ids']
            count = int(cell['arm'].rsplit('k', 1)[1])
            if len(candidate_rows) != 8 or any(len(row) != 32 for row in candidate_rows):
                raise ValueError('candidate shape mismatch')
            if any(len(row) != count or len(set(row)) != count or any(type(token) is not int or not 0 <= token < 50257 for token in row)
                   for document in candidate_rows for row in document):
                raise ValueError('invalid or duplicate candidates')
        docs = protocol['selected_document_ids'][cell['mode']]
        if pred['document_ids'] != docs or len(pred['predictions']) != len(docs):
            raise ValueError('document/sample drift')
        source_truth = truths[(cell['seed'], cell['defense'], cell['mode'])]
        totals, exact, control, shuffled, covered = 0, 0, 0, 0, 0
        per_sequence = []
        for i, (doc, tokens) in enumerate(zip(docs, pred['predictions'])):
            truth = source_truth[doc]['truth']
            if any(type(token) is not int or not 0 <= token < 50257 for token in tokens):
                raise ValueError('invalid predicted token IDs')
            if len(tokens) != 32 or len(truth) != 32:
                raise ValueError('sequence length drift')
            correct = sum(a == b for a, b in zip(tokens, truth))
            shuffled_correct = sum(a == b for a, b in zip(tokens, source_truth[docs[(i - 1) % len(docs)]]['truth']))
            totals += correct
            exact += correct == len(tokens)
            control += source_truth[doc]['control_correct']
            shuffled += shuffled_correct
            per_sequence.append({'document_id': doc, 'correct': correct, 'shuffled_correct': shuffled_correct, 'n_tokens': 32})
            if 'candidate_ids' in pred['diagnostics']:
                covered += sum(token in candidates for token, candidates in zip(truth, pred['diagnostics']['candidate_ids'][i]))
        n_tokens = len(docs) * 32
        recomputed = {'accuracy': totals / n_tokens, 'exact_sequence_match': exact / len(docs),
                      'majority_accuracy': control / n_tokens, 'shuffled_pair_accuracy': shuffled / n_tokens,
                      'n_sequences': len(docs), 'n_tokens': n_tokens, 'per_sequence': per_sequence}
        if any(cell[k] != value for k, value in recomputed.items()):
            raise ValueError('scores do not reproduce from sealed pilot truth')
        cell['candidate_oracle_ceiling'] = covered / n_tokens if 'candidate_ids' in pred['diagnostics'] else None
        if cell['candidate_oracle_ceiling'] is not None and cell['accuracy'] > cell['candidate_oracle_ceiling']:
            raise ValueError('recovery exceeds candidate ceiling')
        cell['depth'] = depth
    return raw, anchor


def plots(summary, output, defense):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = [('ridge_forward', 'Ridge forward', 'o', '#777777'),
              ('direction_backward', 'Gradient direction', 'x', '#a87035'),
              ('forward_beam_', 'Forward beam', 's', '#267a9c'),
              ('joint_union_', 'Joint union', 'D', '#238b50'),
              ('legacy_joint_', 'Legacy joint', '^', '#b45a64'),
              ('sipit_', 'SIPIT (bounded)', '*', '#7557a4')]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True, layout='constrained')
    for row, mode in enumerate(['training', 'inference']):
        for col, depth in enumerate([3, 6]):
            ax = axes[row, col]
            subset = [r for r in summary if r['defense'] == defense and r['mode'] == mode and r['depth'] == depth]
            for prefix, label, marker, color in styles:
                points = sorted([r for r in subset if r['arm'].startswith(prefix)], key=lambda r: r['attack_seconds_per_document'])
                if not points:
                    continue
                ax.plot([r['attack_seconds_per_document'] for r in points], [r['accuracy_pct_mean'] for r in points],
                        marker=marker, color=color, label=label, markersize=7)
            ax.set_xscale('log')
            ax.set_ylim(0, 103)
            ax.grid(alpha=.18)
            ax.set_title(f'{mode.capitalize()} · split {depth}')
            ax.set_xlabel('Amortized attack seconds / document (log scale)')
            if col == 0:
                ax.set_ylabel('Position-aligned recovery (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=3, frameon=False)
    fig.suptitle(f'Recovery versus measured compute · {defense}\nFive-seed means; public setup excluded\nBeam K=8→32; SIPIT 32→128 iterations/token', fontsize=12)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(output / f'recovery-compute-{defense}.{ext}', dpi=180)
    plt.close(fig)


def analyze(root, source_root, protocol_path):
    protocol = json.loads(protocol_path.read_text())
    fidelity_path = root / 'fidelity.json'
    ok, problems = verify_evidence(str(fidelity_path) + '.evidence.json')
    if not ok:
        raise ValueError(f'published-method fidelity evidence failed: {problems}')
    fidelity = json.loads(fidelity_path.read_text())
    if (fidelity['status'] != 'COMPLETE' or len(fidelity['cases']) != 10
            or fidelity['reference_commit'] != protocol['published_baseline']['author_commit']
            or fidelity['implementation_sha256'] != file_sha256(Path(__file__).resolve().parents[1] / 'attacker/reconstruction_baselines.py')):
        raise ValueError('published-method fidelity source/case mismatch')
    fidelity_manifest = json.loads(Path(str(fidelity_path) + '.evidence.json').read_text())
    records, setups, anchors, runtime_inspections = [], {}, {}, {}
    for depth in protocol['depths']:
        raw, anchor = verify_comparison(root / f'split{depth}', protocol, source_root)
        if raw['protocol_sha256'] != file_sha256(protocol_path):
            raise ValueError('protocol bytes changed')
        runtime = json.loads((root / f'runtime-split{depth}.json').read_text())
        if runtime['script_sha256'] != file_sha256(Path(__file__).with_name('inspect_baseline_runtime.py')):
            raise ValueError('runtime inspection script mismatch')
        if runtime['imported']['torch']['version'] != raw['runtime']['torch'] or runtime['torch_cuda'] != raw['runtime']['cuda']:
            raise ValueError('runtime inspection differs from execution runtime')
        runtime_inspections[str(depth)] = runtime
        records.extend(raw['results'])
        anchors[str(depth)] = anchor
        setups.update({(depth, row['id']): row for row in raw['setup']})
    grouped = {}
    for cell in records:
        grouped.setdefault((cell['depth'], cell['defense'], cell['mode'], cell['arm']), []).append(cell)
    summary, seeds = [], []
    for (depth, defense, mode, arm), cells in sorted(grouped.items()):
        accuracy = [r['accuracy'] * 100 for r in cells]
        cold = []
        for r in cells:
            setup = setups[(depth, r['setup_id'])]
            applicable = setup['model_load_and_warmup_seconds']
            if r['uses_ridge_calibration']:
                applicable += setup['public_calibration']['wall_seconds']
            cold.append(applicable + r['work']['wall_seconds'])
            seeds.append({k: r[k] for k in ['depth', 'seed', 'defense', 'mode', 'arm', 'accuracy', 'exact_sequence_match', 'n_tokens', 'n_sequences']} | r['work'])
        summary.append({'depth': depth, 'defense': defense, 'mode': mode, 'arm': arm,
                        'accuracy_pct_mean': statistics.mean(accuracy), 'accuracy_pct_sd': statistics.stdev(accuracy),
                        'majority_pct': statistics.mean(r['majority_accuracy'] * 100 for r in cells),
                        'exact_sequence_match_pct': statistics.mean(r['exact_sequence_match'] * 100 for r in cells),
                        'attack_seconds_per_batch': statistics.mean(r['work']['wall_seconds'] for r in cells),
                        'attack_seconds_per_document': statistics.mean(r['work']['wall_seconds'] / r['n_sequences'] for r in cells),
                        'cold_start_plus_attack_seconds': statistics.mean(cold),
                        'head_forward_token_positions': statistics.mean(r['work']['head_forward_token_positions'] for r in cells),
                        'head_backward_token_positions': statistics.mean(r['work']['head_backward_token_positions'] for r in cells),
                        'vocabulary_dot_products': statistics.mean(r['work']['vocabulary_dot_products'] for r in cells),
                        'peak_allocated_mib': (statistics.mean(r['work']['peak_allocated_bytes'] / 2**20 for r in cells) if all(r['work']['peak_allocated_bytes'] is not None for r in cells) else None),
                        'candidate_oracle_ceiling_pct': statistics.mean(r['candidate_oracle_ceiling'] * 100 for r in cells) if cells[0]['candidate_oracle_ceiling'] is not None else None})
    lookup = {(r['depth'], r['defense'], r['mode'], r['arm'], r['seed']): r for r in records}
    contrasts = []
    matched_checks = 0
    for depth, defense in itertools.product(protocol['depths'], protocol['defenses']):
        for count in protocol['candidate_counts']:
            for baseline in [f'forward_beam_k{count}', 'direction_backward']:
                left = [lookup[(depth, defense, 'training', f'joint_union_k{count}', seed)] for seed in protocol['seeds']]
                right = [lookup[(depth, defense, 'training', baseline, seed)] for seed in protocol['seeds']]
                if baseline.startswith('forward'):
                    for a, b in zip(left, right):
                        fields = ['head_forward_calls', 'head_forward_sequences', 'head_forward_token_positions', 'head_backward_calls']
                        if any(a['work'][f] != b['work'][f] for f in fields):
                            raise ValueError('matched prefix compute drift')
                        matched_checks += 1
                delta = interval([100 * (a['accuracy'] - b['accuracy']) for a, b in zip(left, right)], protocol['statistics']['family_size'])
                contrasts.append({'depth': depth, 'defense': defense, 'mode': 'training', 'attack': f'joint_union_k{count}', 'baseline': baseline, **delta})
        for mode in protocol['modes']:
            left = [lookup[(depth, defense, mode, 'sipit_steps128', seed)] for seed in protocol['seeds']]
            right = [lookup[(depth, defense, mode, 'forward_beam_k32', seed)] for seed in protocol['seeds']]
            delta = interval([100 * (a['accuracy'] - b['accuracy']) for a, b in zip(left, right)], protocol['statistics']['family_size'])
            contrasts.append({'depth': depth, 'defense': defense, 'mode': mode, 'attack': 'sipit_steps128', 'baseline': 'forward_beam_k32', **delta})
    if len(contrasts) != protocol['statistics']['family_size']:
        raise ValueError('comparison family mismatch')
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'seed_metrics.csv', seeds)
    write_csv(output / 'paired_contrasts.csv', contrasts)
    joint_forward = [r for r in contrasts if r['attack'].startswith('joint_union') and r['baseline'].startswith('forward_beam')]
    positive = sum(r['lower'] > 0 for r in joint_forward)
    negative = sum(r['upper'] < 0 for r in joint_forward)
    inconclusive = len(joint_forward) - positive - negative
    sipit_rows = [r for r in summary if r['arm'] == 'sipit_steps128']
    by_group = {(r['depth'], r['defense'], r['mode'], r['arm']): r for r in summary}
    ratios = [r['attack_seconds_per_batch'] / by_group[(r['depth'], r['defense'], r['mode'], 'forward_beam_k32')]['attack_seconds_per_batch'] for r in sipit_rows]
    union_rows = [r for r in summary if r['arm'] == 'joint_union_k32']
    legacy_gains = [r['accuracy_pct_mean'] - by_group[(r['depth'], r['defense'], r['mode'], 'legacy_joint_k32')]['accuracy_pct_mean'] for r in union_rows]
    lines = ['# Stronger reconstruction baseline comparison', '',
             f'{len(records)} completed cells; all prediction files and source evidence verified. Scores independently recomputed from the sealed pilot truth. {matched_checks} paired forward/joint comparisons have identical public-head prefix work.', '',
             'This is an exploratory replay of eight fixed pilot documents per mode, two depths, two defenses and five seeds. Each document has 32 tokens. The protocol was frozen before new outcomes; the source pilot had already been inspected.', '',
             '## Findings', '',
             f'- At matched prefix-evaluation budgets, joint-minus-forward effects have adjusted intervals above zero in {positive}/{len(joint_forward)} conditions, below zero in {negative}, and including zero in {inconclusive}. These are exploratory conditional estimates.',
             f'- At K=32, union proposals improve joint recovery over the legacy backward-only shortlist by {min(legacy_gains):.2f}–{max(legacy_gains):.2f} percentage points across the four training conditions. This is a descriptive implementation comparison, not evidence that adding observations necessarily helps every decoder.',
             f'- The 128-step SIPIT adaptation recovers {min(r["accuracy_pct_mean"] for r in sipit_rows):.2f}–{max(r["accuracy_pct_mean"] for r in sipit_rows):.2f}% across conditions, using {min(ratios):.2f}–{max(ratios):.2f} times the online wall time of the corresponding K=32 forward beam. The two algorithms have different work budgets; the paired table must not be interpreted as an equal-cost ranking.',
             f'- High token accuracy is not full-prompt recovery: joint K=32 exact recovery of complete 32-token training sequences is {min(r["exact_sequence_match_pct"] for r in union_rows):.1f}–{max(r["exact_sequence_match_pct"] for r in union_rows):.1f}% across conditions. Sensitive-span recovery remains unmeasured.',
             '- These results strengthen the empirical attack baseline. They do not validate a scalar risk score, establish architectural generalization, or demonstrate a novel reconstruction principle.', '',
             '## Access and methods', '',
             'All arms are granted the same frozen public prefix weights, embeddings, sequence lengths and calibration documents. The observation ablation gives forward-only arms activations, backward-only direction decoding gradients, and joint arms both. No private weights, dropout masks or evaluation tokens enter reconstruction functions. Prediction files are persisted before scoring.', '',
             '- **Ridge forward:** original calibration-based activation decoder.',
             '- **Forward beam:** K forward proposals per position, beam width 2, activation residual ranking.',
             '- **Joint union:** forward top K/2 plus backward top K/2, deduplicated and filled from forward proposals to exactly K. Position zero uses forward proposals. Same prefix evaluation budget as forward beam.',
             '- **Legacy joint:** K backward proposals after position zero, reproduced as a diagnostic reference.',
             '- **Direction backward:** shifted negative-gradient/embedding cosine heuristic, public majority at position zero. This is not the complete published TAG algorithm.',
             '- **SIPIT:** independently implemented published sequential continuous/discrete reconstruction, at 32/128 iterations per token. Best verified residual fallback, stale public checkpoint and batching are explicit adaptations. No exact-recovery guarantee is claimed here.', '',
             '## Recovery and measured compute', '',
             'Accuracy is five-seed mean ± SD. Time is measured for a batch of eight documents; dividing by eight gives amortized throughput cost, not single-request latency. Setup-plus-attack cost includes the shared full-prefix loading/warmup and applicable ridge calibration; online cost excludes them. The backward heuristic could use a cheaper embedding-only loader, so shared startup is not its optimized standalone cost. CUDA event elapsed time includes host launch gaps. Counts are executed prefix token positions, not FLOP estimates.', '',
             '| Depth | Defense | Mode | Arm | Recovery % ± SD | Exact sequence % | Online seconds / 8 docs | Shared setup + attack seconds | Peak MiB |',
             '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        peak = f"{r['peak_allocated_mib']:.1f}" if r['peak_allocated_mib'] is not None else 'n/a'
        lines.append(f"| {r['depth']} | {r['defense']} | {r['mode']} | {r['arm']} | {r['accuracy_pct_mean']:.2f} ± {r['accuracy_pct_sd']:.2f} | {r['exact_sequence_match_pct']:.2f} | {r['attack_seconds_per_batch']:.3f} | {r['cold_start_plus_attack_seconds']:.3f} | {peak} |")
    lines += ['', '## Paired effects', '', 'Intervals use the frozen family of 24 comparisons, Bonferroni-adjusted Student t over five seeds, conditional on these documents. Joint/forward beam comparisons match prefix evaluations, with gradient proposal overhead measured. SIPIT/beam comparisons have different compute and must be read alongside the timing curves.', '',
              '| Depth | Defense | Mode | Attack | Baseline | Difference pp [adjusted CI] |', '|---|---|---|---|---|---:|']
    for r in contrasts:
        lines.append(f"| {r['depth']} | {r['defense']} | {r['mode']} | {r['attack']} | {r['baseline']} | {r['mean']:.2f} [{r['lower']:.2f}, {r['upper']:.2f}] |")
    lines += ['', '## Candidate coverage diagnostic', '',
              'An oracle allowed perfect selection within these candidate lists cannot exceed the ceiling below. This scoring-only diagnostic does not tune proposals.', '',
              '| Depth | Defense | Mode | Arm | Candidate ceiling % | Actual recovery % |', '|---|---|---|---|---:|---:|']
    for r in summary:
        if r['candidate_oracle_ceiling_pct'] is not None:
            lines.append(f"| {r['depth']} | {r['defense']} | {r['mode']} | {r['arm']} | {r['candidate_oracle_ceiling_pct']:.2f} | {r['accuracy_pct_mean']:.2f} |")
    lines += ['', '## Fidelity and prior art', '',
              'Ten fidelity cases passed: eight tiny-model fixtures and two pretrained public-calibration prefixes, spanning both depths. Recovered IDs and iteration counts matched the pinned author implementation; the prefix adapter also matched full-model hidden states. The sealed fidelity artifact is linked in analysis.json.', '',
              '[SIPIT paper](https://arxiv.org/abs/2510.15511), Algorithm 1 and Appendix E; [author code](https://github.com/giorgosnikolaou/SIPIT) pinned at `820683156b7257313046a4fb3c492e52519525b7`. `experiments/verify_sipit_reference.py` checks predictions, iteration counts and head equivalence on noiseless fixtures against that external checkout. This establishes kernel fidelity on those fixtures, not reproduction of the published large-model benchmark.', '',
              '[BiSR/SIP](https://arxiv.org/abs/2409.00960) is a relevant additional baseline with public-checkpoint mismatch; its trained SIP inversion model requires a substantially larger auxiliary training study than the pilot calibration sample. The assumption mismatch is explicit: the auxiliary inversion-model training budget is absent from this matrix and BiSR white-box levels differ from these cells, so published BiSR numbers are not comparable cell-for-cell and no cell in this report cites them as such. A pilot-scale, development-only scoped adaptation (disjoint-auxiliary SIP-style inverter plus bidirectional candidate selection) exists as the `bisr-baseline` attack module for machinery development, clearly labeled an adaptation rather than a faithful reproduction. [Hidden No More](https://arxiv.org/abs/2505.18332) provides sequential vocabulary matching, but its proceedings-linked author repository returned 404 during this review.', '',
              '## Runtime provenance correction', '',
              'The frozen manifests used a distribution inventory that overwrote earlier packages with later shadowed installations. They consequently list Transformers 5.15.0 from the shared base environment. Separate post-run inspections using the same task interpreters resolve Transformers 4.57.6 from the task virtual environment, alongside PyTorch 2.11.0+cu128 and SciPy 1.18.1. Imported paths, versions, duplicate-distribution order and module hashes are retained in analysis.json. These inspections are not telemetry from the original running processes. The inventory precedence bug is fixed for future evidence; frozen manifests are preserved.', '',
              '## Limitations', ''] + ['- ' + item for item in protocol['limits']]
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    for defense in protocol['defenses']:
        plots(summary, output, defense)
    result = {'status': 'COMPLETE', 'cells': len(records), 'matched_prefix_checks': matched_checks,
              'protocol_sha256': file_sha256(protocol_path), 'comparison_anchors': anchors,
              'fidelity_manifest_sha256': fidelity_manifest['manifest_sha256'],
              'runtime_inspections': runtime_inspections,
              'summary': summary, 'contrasts': contrasts,
              'files': {p.name: file_sha256(p) for p in output.iterdir() if p.suffix in ['.csv', '.md', '.png', '.pdf', '.svg']}}
    artifact = output / 'analysis.json'
    with EvidenceLogger(artifact) as evidence:
        evidence.event('verified_comparison', {'cells': len(records), 'anchors': anchors})
        artifact.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'status': result['status'], 'cells': len(records), 'report': str(output / 'report.md')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/baseline-strengthening'))
    parser.add_argument('--source-root', type=Path, default=Path('results/pilot-gpt2-wikitext2'))
    parser.add_argument('--protocol', type=Path, default=Path('experiments/baseline_protocol.json'))
    args = parser.parse_args()
    analyze(args.root, args.source_root, args.protocol)


if __name__ == '__main__':
    main()
