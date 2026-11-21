"""Freeze development-selected policies, then score all912questions after blind collection."""

import argparse
import json
from pathlib import Path
import pickle
import shutil

from confirmation_common import (HERE, BASELINES, evidence, load_banks, models, run_policy, verify_fixed_prefix, verify_special_banks,
                                 margins, criteria, read, sha, statistics)
from study import fixed


def freeze(args):
    if args.output.exists():
        raise ValueError('Use a new confirmation directory; never overwrite a freeze')
    if (args.data / 'test-collection.json').exists():
        raise ValueError('Controller test replay already started; cannot reselect')
    fixed_receipt = json.loads((args.data / 'fixed-cache-complete.json').read_bytes())
    if fixed_receipt['questions'] != 912 or fixed_receipt['response_bank_sha256'] != sha(args.data / 'test-observations.jsonl'):
        raise ValueError('Complete and preserve the fixed baseline cache before freezing controllers')
    rows = read(args.data / 'splits/test-questions.jsonl')
    if len(rows) != 912:
        raise ValueError('Preserve the entire prospective912panel')
    fitting = [r for split in ('train', 'dev')
               for r in read(args.data / 'splits' / (split + '-questions.jsonl'))]
    if {r['id'] for r in rows} & {r['id'] for r in fitting}:
        raise ValueError('Fresh/fitting overlap')
    candidates = []
    baselines = None
    modules = ('dynamic_grounded_pomdp', 'binary_pomdp', 'calibrated_pomdp', 'root_medium_pomdp',
               'fusion_pomdp', 'subject_fusion_pomdp', 'mentor_pomdp', 'subject_policy_pomdp',
               'packet_pomdp', 'subject_packet_pomdp')
    for probe in args.probe:
        summary = json.loads((probe / 'summary.json').read_bytes())
        if summary['fresh_data_used']:
            raise ValueError('Test outcomes influenced selection')
        identity = json.loads((probe / 'identity.json').read_bytes())
        module = None
        if 'semantic_code' in identity:
            module = next((name for name in modules
                           if sha((HERE.parent if name == 'dynamic_grounded_pomdp' else HERE) / (name + '.py'))
                           == identity['semantic_code']), None)
            if module is None:
                raise ValueError('Unrecognized development policy source')
        if 'baselines' in summary:
            if baselines is not None and baselines != summary['baselines']:
                raise ValueError('Fixed baseline results differ across development probes')
            baselines = summary['baselines']
        for candidate in json.loads((probe / 'search.json').read_bytes()):
            current = candidate.get('module', module)
            if current not in (*modules, 'medium_pomdp'):
                raise ValueError('Unknown policy implementation')
            protocol = ('full-packet' if current in ('packet_pomdp', 'subject_packet_pomdp') else
                        'mentor-reuse' if current == 'mentor_pomdp' else
                        'root-medium' if current == 'root_medium_pomdp' else
                        'medium' if current == 'medium_pomdp' else 'late' if candidate['late'] else 'original')
            candidates.append({**candidate, 'module': current, 'protocol': protocol, 'probe': str(probe)})
    best = lambda group: max(group, key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute']))
    a = best([r for r in candidates if r['family'] == 'A'])
    b = best([r for r in candidates if r['family'] == 'B'])
    hybrids = [r for r in candidates if r['family'] == 'C' and r['dev']['mean_judge_calls'] > 0
               and r['dev']['mean_compute'] < b['dev']['mean_compute']]
    if not hybrids:
        raise ValueError('No hybrid fits the compute target on development data')
    c = best(hybrids)
    special = (args.packet if c['protocol'] == 'full-packet' else
               args.root if c['protocol'] in ('root-medium', 'mentor-reuse') else args.medium)
    if special is None:
        raise ValueError('Supply the selected root-hybrid training bank with --root')
    selected = {'A': a, 'B': b, 'C': c}
    args.output.mkdir(parents=True)
    (args.output / 'models').mkdir()
    assets, descriptors = {}, {}
    for family in ('A', 'C'):
        candidate = selected[family]
        source = Path(candidate['probe']) / (candidate['model'] + '.pkl')
        model = pickle.loads(source.read_bytes())
        folder = Path(model.classifier.directory)
        recipe = json.loads((folder / 'recipe.json').read_bytes())
        key = sha(folder / 'recipe.json')[:16]
        relative = 'observers/' + key
        if key not in assets:
            shutil.copytree(folder, args.output / relative)
            assets[key] = {'directory': relative, 'source_directory': str(folder),
                           'private_volume': 'srr-semantic-training-20261120',
                           'remote': None if recipe.get('components') else '/' + folder.parent.name + '/full',
                           'components': recipe.get('components'),
                           'fit_inputs': {str(p): sha(p) for name in ('contract.json', 'oof.npz', 'training-summary.json')
                                          if (p := folder.parent / name).exists()},
                           'files': {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*') if p.is_file()},
                           'recipe': recipe}
        model.classifier.directory = relative
        destination = args.output / 'models' / (family + '.pkl')
        destination.write_bytes(pickle.dumps(model, protocol=5))
        descriptors[family] = {'file': 'models/' + family + '.pkl', 'sha256': sha(destination),
                               'probe_model_sha256': sha(source), 'observer': key,
                               'observer_directory': relative}
    highest = max(v['accuracy'] for v in baselines.values())
    payload = {'selected': selected, 'models': descriptors, 'observer_assets': assets,
               'evidence': evidence(args.data), 'data_directory': str(args.data),
               'medium_directory': str(special),
               'medium_training_contract_sha256': sha(special / 'contract.json'),
               'medium_training_bank_sha256': sha(special / 'train.jsonl'),
               'medium_development_bank_sha256': sha(special / 'dev.jsonl') if (special / 'dev.jsonl').exists() else None,
               'questions': 912, 'baselines': BASELINES,
               'fixed_cache_contract_sha256': sha(args.data / 'fixed-cache-contract.json'),
               'fixed_cache_bank_bytes': (args.data / 'test-observations.jsonl').stat().st_size,
               'fixed_cache_bank_sha256': fixed_receipt['response_bank_sha256'],
               'targets': {'interpretation': 'percentage points', 'A_B_above_highest_baseline': .10,
                           'C_above_highest_A_B': .10, 'C_stretch': .15, 'C_compute_below_B': True},
               'development_targets': {'A_plus_10_points': a['dev']['accuracy'] >= highest + .1 - 1e-12,
                                       'B_plus_10_points': b['dev']['accuracy'] >= highest + .1 - 1e-12,
                                       'C_plus_10_points': c['dev']['accuracy'] >= max(a['dev']['accuracy'], b['dev']['accuracy']) + .1 - 1e-12},
               'selection': 'Independent maximum development accuracy A and strongest ordinary B; maximum accuracy C among hybrids cheaper than B. Ties minimize compute. No test outcomes used.',
               'roles': {'A': 'Finite-horizon POMDP with discriminative observations, no LLM reviews.',
                         'B': 'Independent ordinary LLM judge policy; strongest development configuration retained.',
                         'C': 'Finite-horizon POMDP acquiring paid LLM observations by information value.'},
               'budget': 'Same frozen4Bsolver, same initial call, at most two further solver calls. Five unchanged fixed baselines. Full-packet C, if selected, retains one ordinary brief review with128token answer-first repairs. Medium variants explicitly declare their deeper review budget.',
               'raw_data': 'GPQA text/options/reference answers and model rationales remain private in ignored runtime files. Publish only code, hashes, metrics, and sanitized costs/correctness.',
               'validation': 'All912reserved questions and each192/720source panel; paired subject-by-panel bootstrap with best baseline/A/B recomputed each draw.',
               'compute': '(4*all solver tokens +3*all physical LLM review tokens +encoder billions*actual token slots)/3.09. Also report raw tokens/calls and35Btotal-parameter sensitivity.'}
    payload['training_observations'] = {'path': str(args.examples / 'train.jsonl'),
                                        'sha256': sha(args.examples / 'train.jsonl')}
    payload['selected_fit_provenance'] = {family: json.loads((Path(selected[family]['probe']) / 'identity.json').read_bytes())
                                         for family in ('A', 'C')}
    if c['protocol'] == 'mentor-reuse':
        reuse = Path(c['reuse_directory'])
        payload['reuse_bank'] = {'directory': str(reuse),
                                 'files': {name: sha(reuse / name) for name in ('contract.json', 'train.jsonl', 'dev.jsonl')}}
    (args.output / 'frozen.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'selected': selected, 'targets': payload['development_targets'],
                      'freeze_sha256': sha(args.output / 'frozen.json')}), flush=True)


def score(args):
    frozen = json.loads((args.output / 'frozen.json').read_bytes())
    if frozen['evidence'] != evidence(args.data):
        raise ValueError('Changed frozen sources/development evidence')
    verify_fixed_prefix(frozen, args.data)
    verify_special_banks(frozen, args.medium)
    completion = json.loads((args.data / 'test-collection.json').read_bytes())
    if completion['freeze_sha256'] != sha(args.output / 'frozen.json') or completion['completed'] != 912:
        raise ValueError('Complete every frozen question before scoring')
    for field, path in (('neural_cache_sha256', args.output / 'test-neural-cache.jsonl'),
                        ('base_sha256', args.data / 'test-observations.jsonl'),
                        ('late_sha256', args.data / 'test-late-observations.jsonl'),
                        ('medium_sha256', args.medium / 'test.jsonl')):
        if completion[field] != sha(path):
            raise ValueError('Changed completed test bank')
    reuse = Path(frozen['reuse_bank']['directory']) if 'reuse_bank' in frozen else None
    if reuse is not None and completion['reuse_sha256'] != sha(reuse / 'test.jsonl'):
        raise ValueError('Changed completed mentor-reuse bank')
    rows, trees, original, late, deeper, deep_reviews = load_banks(
        args.data, 'test', args.medium, frozen['selected']['C']['protocol'], reuse)
    planners = models(frozen, args.output)
    results = {name: [fixed(r, trees[r['id']], *setting) for r in rows] for name, setting in BASELINES.items()}
    for family in ('A', 'B', 'C'):
        results[family] = [run_policy(family, r, trees[r['id']], original, late, deeper[r['id']], deep_reviews,
                                      planners.get(family), frozen['selected'][family]) for r in rows]
    selection = json.loads((args.data / 'splits/selection.json').read_bytes())
    panel_by_id = {qid: panel for panel, ids in selection['strata'].items() for qid in ids}
    panels = {'joint': list(range(len(rows))),
              **{panel: [i for i, r in enumerate(rows) if panel_by_id[r['id']] == panel]
                 for panel in selection['strata']}}
    report = {}
    for panel, indices in panels.items():
        subset = [rows[i] for i in indices]
        outcomes = {k: [v[i] for i in indices] for k, v in results.items()}
        measured = margins(subset, outcomes, panel_by_id)
        checks = criteria(measured)
        stats = {k: statistics(v) for k, v in outcomes.items()}
        report[panel] = {'questions': len(indices), 'methods': stats, 'margins': measured,
                         'criteria': checks, 'highest_baseline': max(BASELINES, key=lambda k: (stats[k]['accuracy'], -stats[k]['mean_compute'])),
                         'all_required_passed': all(v for k, v in checks.items() if k != 'C_plus_15_points_stretch')}
    destination = args.output / 'test'
    if destination.exists():
        raise ValueError('Test already scored; do not overwrite')
    destination.mkdir()
    (destination / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    # These records contain correctness, costs and decisions, but no question,
    # answer string, response text, rationale, options, or retrieval content.
    with (destination / 'per-question.jsonl').open('w') as stream:
        for family, items in results.items():
            for row, item in zip(rows, items):
                stream.write(json.dumps({'policy': family, 'panel': panel_by_id[row['id']], **item}) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze', 'score'))
    for name in ('data', 'medium', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--probe', type=Path, nargs='+')
    p.add_argument('--root', type=Path)
    p.add_argument('--packet', type=Path)
    p.add_argument('--examples', type=Path, default=Path('runs/controller-semantic-hillclimb/grounded-examples'))
    args = p.parse_args()
    (freeze if args.phase == 'freeze' else score)(args)
