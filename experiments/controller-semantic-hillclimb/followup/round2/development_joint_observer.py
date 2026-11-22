"""Gate a completed joint-observer fit into the existing full256 DEV comparison.

This file alone performs no run. Registration requires the passed TRAIN screen
and completed full fit. The128 science questions receive three paid-prefix
observer cache rows each; exact selected-path inference charges remain intact.
"""
import argparse
import copy
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

import development_expanded1280_completion as previous
import fit_joint_observer_pomdp as trained
import joint_observer_pomdp as runtime

io, fixed = trained.io, trained.fixed
original = previous.original
SCHEMA = 'screened-joint-observer-full256-development-v1'


class CachedObserver(runtime.Observer):
    def values(self, row, path, paid):
        entry = self.entry(row, path, paid); key = row['id'], path
        if key not in self.saved or self.saved[key]['entry_sha256'] != fixed.fingerprint(entry):
            raise ValueError('Cached joint observation is absent or its paid input changed')
        value = self.saved[key]
        if value['valid'] != bool(paid[path]['answer']):
            raise ValueError('Current validity changed')
        return np.asarray(value['probability']), value['token_slots']


def verified_fit(folder):
    spec = io.read(folder/'registration.json'); done = io.read(folder/'complete.json')
    summary = io.read(folder/'summary.json')
    if (spec['schema'] != trained.SCHEMA or done['schema'] != trained.SCHEMA
            or done['registration_sha256'] != io.sha(folder/'registration.json')
            or not summary['full_model_fitted'] or summary['development_or_test_read']
            or summary['cloud_calls'] or set(done['models']) != {'original', 'clustered'}
            or done['models'] != summary['models']):
        raise ValueError('Require sealed TRAIN-only full joint-observer fit')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    io.verify({str(folder/n): h for n, h in done['digests'].items()})
    screen = Path(spec['paths']['screen'])
    screen_done = io.read(screen/'complete.json'); screen_summary = io.read(screen/'summary.json')
    if not screen_done['screen_passed'] or not screen_summary['screen']['passed']:
        raise ValueError('Joint-observer TRAIN screen did not pass')
    models = {}
    for name, identity in done['models'].items():
        path = folder/(name+'.pkl')
        if io.sha(path) != identity['sha256']: raise ValueError('Joint POMDP changed')
        model = pickle.loads(path.read_bytes())
        if type(model) is not runtime.Planner or model.fit_receipt != identity['fit_receipt']:
            raise ValueError('Wrong joint POMDP architecture or fitting receipt')
        model.classifier.metadata(); models[name] = model
    evidence = {**spec['input_evidence'], **{str((folder/n).resolve()): h for n, h in done['digests'].items()},
        str((folder/'complete.json').resolve()): io.sha(folder/'complete.json')}
    return models, evidence


def inputs(args):
    # Verify the training gate before decoding any development record.
    models, evidence = verified_fit(args.models)
    spec = io.read(args.previous/'registration.json'); done = io.read(args.previous/'complete.json')
    if (spec['schema'] != previous.SCHEMA or done['schema'] != previous.SCHEMA
            or done['registration_sha256'] != io.sha(args.previous/'registration.json') or done['test_used']):
        raise ValueError('Require the completed original expanded-data development comparison')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    io.verify({str(args.previous/n): h for n, h in done['artifacts'].items()})
    old = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()})
    data, _, values, parent_path, expected = previous.inputs(old)
    if expected != spec: raise ValueError('Changed retained development inputs')
    summary = io.read(args.previous/'summary.json'); retained_folder = old.retain_a
    retained_summary = io.read(retained_folder/'summary.json')
    if summary['selected_A'] != retained_summary['selected_A']:
        raise ValueError('This comparison must retain the strongest completed A')
    rows, trees = data[:2]
    entries = []
    observer = models['original'].classifier
    for row in rows:
        if row['subject'] == 'math': continue
        for path in ('', 'repair', 'retry'):
            entry = observer.entry({**row, 'answer': ''}, path, {p: trees[row['id']][p] for p in ('', path)})
            entries.append({'id': row['id'], 'path': path, 'entry_sha256': fixed.fingerprint(entry),
                'valid': bool(trees[row['id']][path]['answer'])})
    if len(entries) != 384: raise ValueError('All128 science questions and three prefixes required')
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**evidence, **spec['input_evidence'],
            **{str((args.previous/n).resolve()): h for n, h in done['artifacts'].items()},
            str((args.previous/'complete.json').resolve()): io.sha(args.previous/'complete.json')},
        'paths': {k: str(getattr(args, k).resolve()) for k in ('previous', 'models', 'output')},
        'retained_folder': str(retained_folder.resolve()), 'retained_A': summary['selected_A'],
        'dev_questions_sha256': spec['dev_questions_sha256'], 'paid_entries': entries,
        'device': args.device, 'new_neural_calls': 384, 'new_cloud_calls': 0,
        'penalties': list(original.PENALTIES), 'posterior_weights': [0., .5],
        'candidate_count': 32, 'variants': ['original', 'clustered'],
        'math_route': 'Exactly retained A153 math decisions and charges',
        'same_five_baselines_and_matched_controls_required': True,
        'selection': 'Highest full256 accuracy, then lowest compute, retaining prior best A',
        'test_used': False, 'fresh_validation': False,
        'inference_charges': 'Joint observer only at actual visited decision prefixes; no old science encoder call'}
    return data, models, values, retained_summary, record


def retained_items(data, values, retained_summary, folder):
    rows, trees, _, old_observer, _, _, bundle = data
    cached = previous.previous.CachedDense(old_observer.directory, old_observer.temperature)
    cached._recipe = old_observer._recipe; cached._retriever = old_observer._retriever
    cached._backgrounds = old_observer._backgrounds
    cached.expected = {(v['id'], v['path']): v['input_sha256'] for v in bundle['metadata']}
    cached.saved = {(v['id'], v['path']): (*[values[k][i] for k in ('state', 'plain', 'guided')], int(values['token_slots'][i]))
        for i, v in enumerate(bundle['metadata'])}
    cached.joint = {(v['id'], v['path']): values['joint256'][i] for i, v in enumerate(bundle['metadata'])}
    retained = retained_summary['selected_A']; deploy = pickle.loads(Path(retained['model_path']).read_bytes())
    model = copy.deepcopy(deploy); model.math.classifier = old_observer; model.science.classifier = cached
    with original.guarded.no_model_calls():
        items = [previous.retained_math.policy.evaluate(r, trees[r['id']], {}, model, {}, 'A') for r in rows]
    if items != io.read(folder/'private-candidate-items.json')[retained['key']]:
        raise ValueError('Retained A decisions or costs changed')
    math = {r['id']: v for r, v in zip(rows, items) if r['subject'] == 'math'}
    if original.statistics(list(math.values())) != retained_summary['math_metrics']:
        raise ValueError('Retained math changed')
    return deploy, items, math


def observations(args, rows, trees, model, record):
    path = args.output/'private-observations.jsonl'
    saved = [io.json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []
    if len(saved) > 384: raise ValueError('Extra observation record')
    recipe = model.classifier.metadata()
    for i, value in enumerate(saved):
        if ({k: value[k] for k in record['paid_entries'][i]} != record['paid_entries'][i]
                or value['recipe_sha256'] != model.classifier.recipe_sha256
                or value['parameters'] != recipe['parameters']):
            raise ValueError('Changed saved observation identity')
        p = np.asarray(value['probability'])
        if (p.shape != (16,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1.)
                or type(value['token_slots']) is not int or not 0 < value['token_slots'] <= 512
                or (not value['valid'] and p[8:].any())):
            raise ValueError('Invalid saved probability or compute charge')
    observer = copy.deepcopy(model.classifier); observer.device = args.device
    by_id = {r['id']: r for r in rows}
    from collect_auxiliary_second_pairs import durable_line
    with path.open('a') as stream:
        for meta in record['paid_entries'][len(saved):]:
            qid, prefix = meta['id'], meta['path']; row = {**by_id[qid], 'answer': ''}
            p, slots = observer.values(row, prefix, {v: trees[qid][v] for v in ('', prefix)})
            value = {**meta, 'probability': p.tolist(), 'token_slots': slots,
                'recipe_sha256': observer.recipe_sha256, 'parameters': recipe['parameters']}
            durable_line(stream, value); saved.append(value)
            if len(saved) % 32 == 0: print({'completed_neural_rows': len(saved), 'total': 384}, flush=True)
    cached = CachedObserver(observer.directory, observer.input_builder, observer.recipe_sha256)
    cached.saved = {(v['id'], v['path']): v for v in saved}
    if len(cached.saved) != 384: raise ValueError('Duplicate observation key')
    return cached


def evaluate(args, data, models, values, retained_summary, record):
    rows, trees = data[:2]
    deploy_parent, old_items, math = retained_items(data, values, retained_summary, Path(record['retained_folder']))
    cached = observations(args, rows, trees, models['original'], record)
    with original.guarded.no_model_calls():
        # Reuse the original control builder, which checks all original baseline
        # metrics against their retained evidence and computes matched controls.
        baselines, matched = previous.previous.controls(rows, trees, math, retained_summary)
        candidates = []; all_items = {}
        for family, deploy_model in models.items():
            model = copy.deepcopy(deploy_model); model.classifier = cached
            for weight in (0., .5):
                groups = [[math[r['id']] if r['subject'] == 'math' else runtime.evaluate_science(r, trees[r['id']], {}, model,
                    {'penalty': penalty, 'posterior_weight': weight}, 'A') for r in rows] for penalty in original.PENALTIES]
                chosen = {s: max(range(7), key=lambda i: original.fit_rank({'dev': original.statistics(
                    [v for r, v in zip(rows, groups[i]) if r['subject'] == s])})) for s in ('chemistry', 'physics')}
                groups.append([math[r['id']] if r['subject'] == 'math' else groups[chosen[r['subject']]][i] for i, r in enumerate(rows)])
                for index, items in enumerate(groups):
                    settings = {'penalty': original.PENALTIES[index], 'posterior_weight': weight} if index < 7 else {
                        'by_subject': {s: {'penalty': original.PENALTIES[i], 'posterior_weight': weight} for s, i in chosen.items()}}
                    deploy = runtime.SubjectPlanner(copy.deepcopy(deploy_parent.math), copy.deepcopy(deploy_model),
                        deploy_parent.math_settings, settings, {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
                            'retained_A_sha256': record['retained_A']['model_sha256'], 'math_unchanged': True})
                    key = family+'-w'+str(weight)+'-'+str(index); path = args.output/(key+'.pkl')
                    io.atomic_bytes(path, pickle.dumps(deploy, protocol=5))
                    candidates.append({'key': key, 'family': family, 'model_path': str(path.resolve()),
                        'model_sha256': io.sha(path), 'policy_module': 'joint_observer_pomdp', 'settings': {},
                        'math_settings': deploy.math_settings, 'science_settings': settings, 'dev': original.statistics(items)})
                    all_items[key] = items
    selected = max([record['retained_A'], *candidates], key=original.fit_rank)
    result = {'schema': SCHEMA, 'selected_A': selected, 'candidates': candidates, 'retained_A': record['retained_A'],
        'baselines': baselines, 'matched_complete_controls': matched, 'retained_A_exactly_reproduced': True,
        'development_questions': 256, 'training_questions': 1280, 'new_neural_calls': 384, 'new_cloud_calls': 0,
        'test_used': False, 'fresh_validation': False, 'registration_sha256': args.registration_sha256,
        'A_above_highest_baseline': selected['dev']['accuracy']-max(v['accuracy'] for v in baselines.values())}
    io.write(args.output/'private-candidate-items.json', all_items); io.write(args.output/'summary.json', result)
    io.write(args.output/'identity.json', {'schema': SCHEMA, 'selected_A': selected,
        'dev_questions_sha256': record['dev_questions_sha256'], 'registration_sha256': args.registration_sha256,
        'test_used': False, 'policy_module': selected['policy_module']})
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'test_used': False, 'artifacts': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})
    print({'selected_A': selected['dev'], 'best_new': max(candidates, key=original.fit_rank)['dev']}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=('register', 'evaluate'))
    for name in ('previous', 'models', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps'); parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    with previous.previous.parent_io.no_test_reads(), threadpool_limits(limits=1):
        with original.guarded.no_model_calls(): bundle = inputs(args)
        record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed development registration')
        with io.lock(args.output/'.run.lock'):
            if (args.output/'complete.json').exists(): raise ValueError('Preserve completed comparison')
            evaluate(args, *bundle)


if __name__ == '__main__': main()
