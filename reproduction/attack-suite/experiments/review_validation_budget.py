"""Estimate validation online GPU occupancy from measured, already observed baselines.

Reads only protocol and old summary CSV. Does not load evaluation data or run attacks.
The scenarios are engineering planning estimates, not statistical confidence limits.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(protocol, rows, documents):
    if documents < protocol['minimum_documents']:
        raise ValueError('documents below protocol minimum')
    if protocol['attack_batch_size'] != 8 or protocol['sequence_length'] != 32:
        raise ValueError('source timing measurements require batch 8 and sequence length 32')
    measured = {}
    for row in rows:
        key = (int(row['depth']), row['mode'], row['arm'])
        measured.setdefault(key, []).append(float(row['attack_seconds_per_batch']))
    repeats = len(protocol['victim_seeds']) * len(protocol['attack_seeds']) * len(protocol['knowledge'])
    batches = math.ceil(documents / 8)
    reuse = protocol.get('execution', {}).get('reuse_identical_activation_modes', False)
    if type(reuse) is not bool:
        raise ValueError('reuse_identical_activation_modes must be boolean')
    spec = protocol.get('capture_defenses') or {}
    defended = spec.get('defended', [])
    if not isinstance(defended, list) or len(defended) != len(set(defended)) or \
            any(type(x) is not str or not x or x == 'none' or '@' in x for x in defended):
        raise ValueError('invalid capture_defenses axis')
    output = []
    for depth in protocol['depths']:
        for mode in ('training', 'inference'):
            specifications = [(f'forward_k{k}', 'forward', k, None) for k in protocol['candidate_counts']]
            specifications += [(f'sipit_steps{s}', 'sipit', s, None) for s in protocol['sipit_steps']]
            if mode == 'training':
                specifications.append(('backward', 'backward', 0, None))
                specifications += [(f'joint_{c}_k{k}', 'joint', k, None) for k in protocol['candidate_counts']
                                   for c in protocol['gradient_conditions'] + ['random']]
                for defense in defended:
                    specifications += [(f'forward_k{k}@{defense}', 'forward', k, defense)
                                       for k in protocol['candidate_counts']]
                    specifications += [(f'sipit_steps{s}@{defense}', 'sipit', s, defense)
                                       for s in protocol['sipit_steps']]
            for arm, family, budget, defense in specifications:
                if family == 'backward':
                    timings = measured[(depth, mode, 'direction_backward')]
                    basis = 'measured backward heuristic'
                else:
                    prefix = {'forward': 'forward_beam_k', 'joint': 'joint_union_k', 'sipit': 'sipit_steps'}[family]
                    key = (depth, mode, prefix + str(budget))
                    if key in measured:
                        timings = measured[key]
                        basis = 'measured ' + prefix + str(budget)
                    else:
                        lo, hi = ((32, 128) if family == 'sipit' else (8, 32))
                        if not lo <= budget <= hi:
                            raise ValueError('budget outside measured interpolation range')
                        a, b = measured[(depth, mode, prefix + str(lo))], measured[(depth, mode, prefix + str(hi))]
                        if len(a) != len(b):
                            raise ValueError('inconsistent source measurement groups')
                        fraction = (budget-lo)/(hi-lo)
                        timings = [x+(y-x)*fraction for x, y in zip(a,b)]
                        basis = f'linear interpolation {prefix}{lo} to {prefix}{hi}'
                mean = sum(timings)/len(timings)
                reused = reuse and mode == 'inference' and family in ('forward', 'sipit')
                nominal_hours = mean*batches*repeats/3600
                output.append({'depth': depth, 'mode': mode, 'arm': arm, 'family': family,
                               'defense': defense,
                               'cells': repeats, 'physical_cells': 0 if reused else repeats,
                               'reuse_training_execution': reused, 'batches_per_cell': batches,
                               'source_seconds_per_batch_min': min(timings),
                               'source_seconds_per_batch_mean': mean,
                               'source_seconds_per_batch_max': max(timings),
                               'nominal_all_executed_gpu_hours': nominal_hours,
                               'estimated_online_gpu_hours': 0.0 if reused else nominal_hours,
                               'timing_basis': basis if defense is None else
                                               basis + '; defended capture, identical attack-side online work'})
    total = sum(r['estimated_online_gpu_hours'] for r in output)
    nominal = sum(r['nominal_all_executed_gpu_hours'] for r in output)
    physical_cells = sum(r['physical_cells'] for r in output)
    assumptions = [
        'Source summary reports five-seed means per eight-document batch on NVIDIA GB10; average the available none/Gaussian defense means.',
        'New protocol uses the same architecture, sequence length and batch size as source timing measurements.',
        'When reuse_identical_activation_modes is true, inference forward/SIPIT rows reference the matching training execution with identical observations, head, arm and seed. Logical coverage remains unchanged; no second timing measurement is claimed.',
        'Nominal all-executed estimates use the original training/inference timings; effective estimates retain training timing and remove inference duplicate execution. Old training/inference differences are not present in the new paired probe.',
        'Scale linearly with batch count; a partial final batch is conservatively treated as full.',
        'Interpolate unmeasured K16 and SIPIT64 between measured budgets, without claiming exact linear runtime.',
        'Use joint true-gradient timings for shuffled, random, noisy and quantized controls; transformation and sampling overhead is unmeasured.',
        'Use the same timing estimate for exact/public head access and all new victim seeds; exact-head SIPIT may terminate sooner.',
        'Totals are online measured wall-time occupancy, not GPU busy time, FLOPs, or elapsed time on multiple devices.',
        '25% and 50% contingencies are explicit planning scenarios, not confidence bounds or guaranteed upper limits.',
        'Exclude model downloads/loading, calibration, training, capture/export, scoring, report serialization, evidence hashing, and statistical analysis; these require separate measurement.',
        'Only ten independently trained victim runs and the selected documents are sampling units. Attack seeds and repeated arms are not independent observations.',
    ]
    if defended:
        assumptions.append('Defended-capture arms run the identical attack code with identical budgets against the post-defense wire, so they reuse the undefended forward/SIPIT per-batch timings; defender training and capture transformation overhead is excluded under the same training/capture exclusion as undefended victim runs.')
    return {
        'documents': documents, 'cells': sum(r['cells'] for r in output),
        'document_reconstructions': documents*sum(r['cells'] for r in output),
        'batches': batches*sum(r['cells'] for r in output),
        'reuse_identical_activation_modes': reuse,
        'capture_defenses': defended,
        'physical_cells': physical_cells,
        'physical_document_reconstructions': documents*physical_cells,
        'physical_batches': batches*physical_cells,
        'nominal_all_executed_gpu_hours': nominal,
        'duplicate_execution_savings_gpu_hours': nominal-total,
        'estimated_online_gpu_hours': total,
        'planning_gpu_hours_with_25_percent_contingency': total*1.25,
        'planning_gpu_hours_with_50_percent_contingency': total*1.5,
        'by_depth_gpu_hours': {str(d): sum(r['estimated_online_gpu_hours'] for r in output if r['depth']==d) for d in protocol['depths']},
        'nominal_by_depth_gpu_hours': {str(d): sum(r['nominal_all_executed_gpu_hours'] for r in output if r['depth']==d) for d in protocol['depths']},
        'by_depth_planning_gpu_hours': {str(d): {'25_percent_contingency': sum(r['estimated_online_gpu_hours'] for r in output if r['depth']==d)*1.25, '50_percent_contingency': sum(r['estimated_online_gpu_hours'] for r in output if r['depth']==d)*1.5} for d in protocol['depths']},
        'by_family_gpu_hours': {f: sum(r['estimated_online_gpu_hours'] for r in output if r['family']==f) for f in ('forward','joint','backward','sipit')},
        'rows': output,
        'assumptions': assumptions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('experiments/validation_protocol.json'))
    parser.add_argument('--summary', type=Path, default=Path('results/baseline-strengthening/analysis/summary.csv'))
    parser.add_argument('--documents', type=int, default=256)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be fresh')
    protocol = json.loads(args.protocol.read_text())
    with args.summary.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    result = estimate(protocol, rows, args.documents)
    result['provenance'] = {'protocol_path': str(args.protocol), 'protocol_sha256': sha256(args.protocol),
                            'source_summary_path': str(args.summary), 'source_summary_sha256': sha256(args.summary),
                            'script_sha256': sha256(Path(__file__))}
    args.output.mkdir(parents=True)
    (args.output/'budget.json').write_text(json.dumps(result, indent=2)+'\n')
    with (args.output/'budget.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result['rows'][0]))
        writer.writeheader(); writer.writerows(result['rows'])
    report = [f"# Validation budget review", '',
              f"{result['cells']:,} cells × {args.documents} documents = {result['document_reconstructions']:,} document reconstructions; {result['batches']:,} batches.", '',
              f"Physical executions: **{result['physical_cells']:,} cells**, {result['physical_document_reconstructions']:,} document reconstructions, {result['physical_batches']:,} batches.",
              f"Nominal all-executed occupancy: **{result['nominal_all_executed_gpu_hours']:.1f} GPU-hours**; duplicate execution savings: **{result['duplicate_execution_savings_gpu_hours']:.1f} GPU-hours**.",
              f"Estimated effective online GPU occupancy: **{result['estimated_online_gpu_hours']:.1f} GPU-hours**.",
              f"Planning scenarios with 25% / 50% contingency: **{result['planning_gpu_hours_with_25_percent_contingency']:.1f} / {result['planning_gpu_hours_with_50_percent_contingency']:.1f} GPU-hours**.", '',
              '| Split depth | Nominal GPU-hours | Effective GPU-hours | +25% | +50% |', '|---|---:|---:|---:|---:|']
    report += [f"| {depth} | {result['nominal_by_depth_gpu_hours'][depth]:.1f} | {hours:.1f} | {result['by_depth_planning_gpu_hours'][depth]['25_percent_contingency']:.1f} | {result['by_depth_planning_gpu_hours'][depth]['50_percent_contingency']:.1f} |" for depth, hours in result['by_depth_gpu_hours'].items()]
    report += ['', '| Attack family | Online GPU-hours |', '|---|---:|']
    report += [f'| {family} | {hours:.1f} |' for family, hours in result['by_family_gpu_hours'].items()]
    report += ['', '## Assumptions and limitations', ''] + ['- '+a for a in result['assumptions']]
    report += ['', '## Provenance', '', '```json', json.dumps(result['provenance'], indent=2), '```', '']
    (args.output/'report.md').write_text('\n'.join(report))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','assumptions')}, indent=2))


if __name__ == '__main__':
    main()
