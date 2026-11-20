"""Select genuine POMDP, LLM, and hybrid controllers before a fresh 600-question test."""

import argparse
from functools import lru_cache
import gzip
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from extended_acquire import (CHECKPOINTS, HERE, INITIAL_CAP, JUDGE, JUDGE_ACTIVE, JUDGE_CAP,
                              JUDGE_DIGEST, JUDGE_TOTAL, MODEL, MODEL_DIGEST, PATHS, ROOT,
                              SEED, SOLVER_CAP, digest, judge_prompt, paired_archive,
                              parse_answer, parse_verdict, read, solver_prompt)
from context_pomdp import Planner, evaluate, fit
from hillclimb import new_run, paired, save
from study import describe, evidence as legacy_evidence, fixed, normalized_hash, records as legacy_records


def locate(source, name):
    path = source / name
    return path if path.exists() else path.with_suffix(path.suffix + '.gz')


def records(source, split, complete=True):
    questions = read(source / 'splits' / f'{split}-questions.jsonl')
    manifest = json.loads((source / 'splits/selection.json').read_bytes())['splits'][split]
    if len(questions) != manifest['n'] or digest(source / 'splits' / f'{split}-questions.jsonl') != manifest['sha256']:
        raise ValueError('Question split changed')
    observations = read(locate(source, f'{split}-observations.jsonl'))
    known = {(r['id'], r['kind'], r['path']): r['response'] for r in observations}
    if len(known) != len(observations):
        raise ValueError('Duplicate observations')
    ids = {r['id'] for r in questions}
    allowed = {(i, kind, p) for i in ids for kind, paths in [('solver', PATHS), ('judge', CHECKPOINTS)] for p in paths}
    if not set(known) <= allowed or (complete and set(known) != allowed):
        raise ValueError('Unexpected or missing collection paths')
    trees = {i: {p: known[i, 'solver', p] for p in PATHS if (i, 'solver', p) in known} for i in ids}
    judges = {(i, p): known[i, 'judge', p] for i in ids for p in CHECKPOINTS if (i, 'judge', p) in known}
    for row in questions:
        tree = trees[row['id']]
        if '' not in tree:
            raise ValueError('Missing compact initial response')
        for kind, paths in [('solver', tree), ('judge', [p for i, p in judges if i == row['id']])]:
            for path in paths:
                node = tree[path] if kind == 'solver' else judges[row['id'], path]
                model, fingerprint, cap = (JUDGE, JUDGE_DIGEST, JUDGE_CAP) if kind == 'judge' else (MODEL, MODEL_DIGEST, SOLVER_CAP if path else INITIAL_CAP)
                prompt = judge_prompt(row, path, tree) if kind == 'judge' else solver_prompt(row, path, tree, judges)
                if node['model'] != model or node['model_digest'] != fingerprint or node['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest():
                    raise ValueError(f'Changed inference protocol: {row["id"]}/{kind}/{path}')
                if node['input_tokens'] <= 0 or not 0 <= node['output_tokens'] <= cap:
                    raise ValueError('Invalid token charge')
                if kind == 'judge':
                    if node.get('response_format') != 'text-json':
                        raise ValueError('Changed reviewer output protocol')
                    obj, valid = parse_verdict(node['text'])
                    answer = parse_answer(row['domain'], 'Final: ' + str(obj.get('answer', '')))
                    if (obj, valid, answer, obj['p_correct'], obj['action']) != (node['verdict'], node['parse_valid'], node['answer'], node['p_correct'], node['action']):
                        raise ValueError('Changed review parsing')
                elif parse_answer(row['domain'], node['text']) != node['answer']:
                    raise ValueError('Changed solver parsing')
                if kind == 'judge' or path:
                    logical = path.replace('guided', 'repair')
                    seed = int(hashlib.sha256(f'{SEED}:{row["id"]}:{logical}'.encode()).hexdigest()[:8], 16) % 2**31
                    if node['seed'] != seed or node['temperature'] != (0 if kind == 'judge' else .65):
                        raise ValueError('Changed decoding settings')
    return questions, trees, judges


@lru_cache(maxsize=8)
def training_archive(source):
    forbidden = {r['id'] for split in ('train', 'dev', 'test')
                 for r in read(source / 'splits' / f'{split}-questions.jsonl')}
    roots = {r['id']: r for r in paired_archive()}
    legacy = ROOT / 'results/controller-hillclimb/pomdp-abc/data'
    for split in ('train', 'dev', 'test'):
        rows, trees, _ = legacy_records(legacy, split)
        roots.update({r['id']: {**r, 'compact': trees[r['id']]['']} for r in rows})
    return [r for key, r in sorted(roots.items()) if key not in forbidden]


def evidence(source):
    result = {'code/' + p.name: digest(p) for p in sorted(HERE.glob('*.py'))}
    result.update({'split/' + p.name: digest(p) for p in sorted((source / 'splits').glob('*.json*'))})
    result.update({f'{s}-observations': normalized_hash(source / f'{s}-observations.jsonl') for s in ('train', 'dev')})
    result['training_archive'] = hashlib.sha256(json.dumps(training_archive(source), sort_keys=True).encode()).hexdigest()
    result.update({'legacy/' + k: v for k, v in legacy_evidence(ROOT / 'results/controller-hillclimb/pomdp-abc/data').items()})
    return result


def statistics(items):
    result = describe(items)['all']
    result['mean_total_parameter_compute'] = float(np.mean([r.get('total_parameter_compute', r['compute']) for r in items]))
    return result


def batch(rows, trees, judges, model, settings, method):
    return [evaluate(r, trees[r['id']], judges, model, settings, method) for r in rows]


def select(source, output):
    train, trees, judges = records(source, 'train')
    dev, dev_trees, dev_judges = records(source, 'dev')
    if {r['id'] for r in train} & {r['id'] for r in dev}:
        raise ValueError('Fit and development overlap')
    baselines = {}
    for name, action, depth in [('Stop', 'stop', 0), ('Repair once', 'repair', 1),
                                ('Repair twice', 'repair', 2), ('Retry once', 'retry', 1),
                                ('Retry twice', 'retry', 2)]:
        items = [fixed(r, dev_trees[r['id']], action, depth) for r in dev]
        baselines[name] = {'action': action, 'depth': depth, 'dev': statistics(items)}
    reference = max(baselines, key=lambda k: (baselines[k]['dev']['accuracy'], -baselines[k]['dev']['mean_compute']))
    budget = baselines[reference]['dev']['mean_compute']
    candidates, models = [], {}

    def consider(family, key, settings, items):
        metrics = statistics(items)
        candidate = {'family': family, 'fit': key, 'settings': settings, 'dev': metrics,
                     'within_best_baseline_compute': metrics['mean_compute'] <= budget}
        candidates.append(candidate)

    settings_grid = list(product((0., .015, .04, .08, .15), (.7, 1., 1.4), (False, True), ('binned', 'calibrated')))
    for strength, smoothing, richness in product((32., 128.), (4., 16.), ('linear', 'context')):
        key = f'{strength}:{smoothing}:{richness}'
        model = fit(train, trees, judges, training_archive(source), strength, smoothing, richness)
        models[key] = model
        for penalty, science_weight, contextual, initial_mode in settings_grid:
            settings = {'penalty': penalty, 'science_weight': science_weight, 'contextual': contextual,
                        'initial_mode': initial_mode, 'threshold': 0.}
            consider('A', key, settings, batch(dev, dev_trees, dev_judges, model, settings, 'A'))
            for threshold, reuse in product((-.02, 0., .02, .05), (False, True)):
                current = {**settings, 'threshold': threshold, 'reuse_review': reuse}
                consider('C', key, current, batch(dev, dev_trees, dev_judges, model, current, 'C'))
        print(f'Development fit {key} complete', flush=True)
    for threshold, reuse in product(('action', 'agreement', .5, .8, .95, .98, .995, 1.), (False, True)):
        settings = {'judge_stop': threshold, 'reuse_review': reuse}
        consider('B', None, settings, batch(dev, dev_trees, dev_judges, None, settings, 'B'))
    chosen = {}
    for family in ('A', 'B', 'C'):
        members = [r for r in candidates if r['family'] == family]
        if family == 'C':
            members = [r for r in members if r['dev']['mean_judge_calls'] > 0]
            if not members:
                raise ValueError('No hybrid candidate actually used the LLM on development')
        feasible = [r for r in members if r['within_best_baseline_compute']]
        chosen[family] = max(feasible or members, key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute']))
    c = chosen['C']
    c_results = batch(dev, dev_trees, dev_judges, models[c['fit']], c['settings'], 'C')
    checkpoints = sum(min(r['solver_calls'], 2) for r in c_results)
    rate = sum(r['judge_calls'] for r in c_results)/checkpoints
    payload = {'evidence': evidence(source), 'selected': chosen,
               'models': {k: models[k].as_dict() for k in sorted({r['fit'] for r in chosen.values() if r['fit'] is not None})},
               'baselines': baselines, 'primary_baseline': reference, 'compute_budget': budget,
               'selection_objective': 'Maximize development accuracy among candidates no more expensive than the development-selected strongest fixed baseline; break ties with lower compute. If no candidate in a family is feasible, report its most accurate candidate and its compute excess.',
               'targets': {'A': .65, 'B': .65, 'C': .75},
               'test_rule': 'Targets require accuracy strictly above each threshold on all 600 reserved questions.',
               'architecture': {'A': 'Two-state contextual POMDP with Bayesian filtering and finite-horizon Bellman planning.',
                                'B': 'Frozen LLM action controller; no POMDP used.',
                                'C': 'Contextual POMDP with selective paid LLM observations and reviewed repair.'},
               'solver': {'model': MODEL, 'digest': MODEL_DIGEST, 'initial_cap': INITIAL_CAP, 'continuation_cap': SOLVER_CAP},
               'reviewer': {'model': JUDGE, 'digest': JUDGE_DIGEST, 'active_parameters_billions': JUDGE_ACTIVE, 'total_parameters_billions': JUDGE_TOTAL, 'cap': JUDGE_CAP},
               'compute': 'Sum of full input/output token counts times nominal active parameter counts, divided by 3.09B. Also report total-parameter sensitivity; neither proxy measures memory or wall time.',
               'train_questions': len(train), 'dev_questions': len(dev), 'auxiliary_initial_training': len(training_archive(source)),
               'random_control': {'fit': c['fit'], 'settings': {**c['settings'], 'rate': rate}}}
    with new_run(output) as stage:
        save(stage / 'frozen.json', payload)
        save(stage / 'development-search.json', candidates)
    for family, candidate in chosen.items():
        print(family, json.dumps(candidate), flush=True)
    print('Baselines', json.dumps(baselines), flush=True)


def methods(frozen):
    planners = {k: Planner.restore(v) for k, v in frozen['models'].items()}
    result = [(name, planners.get(c['fit']), c['settings'], name) for name, c in frozen['selected'].items()]
    c, random_control = frozen['selected']['C'], frozen['random_control']
    result += [('C without review', planners[c['fit']], c['settings'], 'never'),
               ('C always reviews', planners[c['fit']], c['settings'], 'always'),
               ('Random reviews', planners[random_control['fit']], random_control['settings'], 'random')]
    return result


def score(source, output):
    frozen = json.loads((output / 'frozen.json').read_bytes())
    if frozen['evidence'] != evidence(source):
        raise ValueError('Source or development evidence changed after freeze')
    rows, trees, judges = records(source, 'test', False)
    metadata = json.loads((source / 'test-collection.json').read_bytes())
    if metadata['completed'] != len(rows) or metadata['freeze_sha256'] != digest(output / 'frozen.json'):
        raise ValueError('The entire frozen holdout must complete before scoring')
    forbidden = {r['id'] for split in ('train', 'dev') for r in read(source / 'splits' / f'{split}-questions.jsonl')}
    forbidden.update(r['id'] for r in training_archive(source))
    if forbidden & {r['id'] for r in rows}:
        raise ValueError('Training IDs entered test')
    results = {name: [fixed(r, trees[r['id']], config['action'], config['depth']) for r in rows]
               for name, config in frozen['baselines'].items()}
    for name, model, settings, method in methods(frozen):
        results[name] = batch(rows, trees, judges, model, settings, method)
    summaries = {name: describe(items) for name, items in results.items()}
    for name, items in results.items():
        summaries[name]['all'].update(statistics(items))
    measurement = {name: {'correct': np.array([r['correct'] for r in items]), 'work': np.array([r['compute'] for r in items])}
                   for name, items in results.items()}
    rng = np.random.default_rng(SEED + 1)
    comparisons = {name: {ref: paired(measurement[name], measurement[ref], rng)
                          for ref in results if ref != name} for name in ('A', 'B', 'C')}
    targets = {name: {'target': threshold, 'accuracy': summaries[name]['all']['accuracy'],
                      'passed': summaries[name]['all']['accuracy'] > threshold,
                      'accuracy_above_primary_baseline': summaries[name]['all']['accuracy'] > summaries[frozen['primary_baseline']]['all']['accuracy'],
                      'compute_below_primary_baseline': summaries[name]['all']['mean_compute'] < summaries[frozen['primary_baseline']]['all']['mean_compute']}
               for name, threshold in frozen['targets'].items()}
    best_test_fixed = max(frozen['baselines'], key=lambda k: (summaries[k]['all']['accuracy'], -summaries[k]['all']['mean_compute']))
    with new_run(output / 'test') as stage:
        save(stage / 'summary.json', {'methods': summaries, 'paired': comparisons, 'targets': targets,
                                     'primary_baseline': frozen['primary_baseline'],
                                     'highest_test_accuracy_fixed_baseline': best_test_fixed,
                                     'comparison_note': 'The primary baseline was selected on development. The highest test-accuracy fixed baseline is a descriptive comparison; no policy is changed after scoring.'})
        with (stage / 'per_question.jsonl').open('w') as stream:
            for name, items in results.items():
                for row in items:
                    stream.write(json.dumps({'policy': name, **row}) + '\n')
    for name, summary in summaries.items():
        stats = summary['all']
        print(f'{name:20} {stats["correct"]}/{stats["n"]} {stats["accuracy"]:.2%} compute={stats["mean_compute"]:.2f}')
    print('Targets', json.dumps(targets))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('select', 'score'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (select if args.phase == 'select' else score)(args.data, args.output)


if __name__ == '__main__':
    main()
