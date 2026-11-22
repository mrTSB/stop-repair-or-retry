"""TRAIN-only learnability of choosing an existing review protocol from root.

This is a routing diagnostic, not a new hybrid controller. All64 previously
registered science questions remain, and every held question is excluded from
outcome/cost estimation. Only the paid root observation is visible at routing.
Earlier aggregate pilot outcomes are known; no fresh-validation claim is made.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import policy_crossfit_diagnostic as data_source
import strong_late_review_pilot as late

io, fixed, nested = data_source.io, data_source.fixed, data_source.nested
SCHEMA = 'paid-root-review-portfolio-crossfit-TRAIN-v1'
PILOTS = ('strong-late-review-pilot-v1', 'long4096-root-review-pilot-v1',
    'blind-first-root-review-pilot-v1', 'grounded1024-root-review-pilot-v2',
    'advanced-grounded1024-root-review-pilot-v1')
PENALTY = .02
RECIPES = ('linear', 'forest')


def pilot_table(root):
    table = {}; evidence = {}; subjects = {}
    for index, name in enumerate(PILOTS):
        directory = root/name; spec = io.read(directory/'registration.json')
        done = io.read(directory/'complete.json'); graded = io.read(directory/'graded/summary.json')
        path = directory/'graded/private-outcomes.json'
        if (not graded['training_only'] or graded['test_used'] or graded['questions'] != 64
                or graded['complete_sha256'] != io.sha(directory/'complete.json')
                or graded['private_outcomes_sha256'] != io.sha(path)
                or done['registration_sha256'] != io.sha(directory/'registration.json')
                or graded['transport_error_records']):
            raise ValueError('Require all sealed complete TRAIN pilots without missing cost')
        io.verify(spec['source_files'])
        io.verify({str(directory/n): h for n, h in done['digests'].items()})
        records = io.read(path)
        if len(records) != 64 or len({r['id'] for r in records}) != 64:
            raise ValueError('Exactly64 unique questions required')
        if index and set(table) != {r['id'] for r in records}:
            raise ValueError('Pilot cohorts differ')
        for row in records:
            qid = row['id']; subjects[qid] = row['subject']; methods = table.setdefault(qid, {})
            for method, value in row['methods'].items():
                if name.startswith('advanced-') and method.startswith('Grounded1024'):
                    method = 'Advanced '+method
                public = {k: value[k] for k in ('correct', 'compute', 'solver_calls', 'logical_review_calls')}
                if methods.get(method, public) != public:
                    raise ValueError('Shared fixed control changed')
                if public['solver_calls'] > 3 or public['logical_review_calls'] > 1:
                    raise ValueError('Portfolio contains an over-budget trajectory')
                methods[method] = public
        for filename in ('registration.json', 'complete.json', 'graded/summary.json', 'graded/private-outcomes.json'):
            evidence[str((directory/filename).resolve())] = io.sha(directory/filename)
        evidence.update({str((directory/n).resolve()): h for n, h in done['digests'].items()})
        evidence.update(spec['source_files'])
    if Counter(subjects.values()) != {'chemistry': 32, 'physics': 32}:
        raise ValueError('The original balanced64 cohort must be unchanged')
    names = sorted(next(iter(table.values())))
    if any(sorted(v) != names for v in table.values()):
        raise ValueError('Every question requires every fixed protocol')
    return table, subjects, names, evidence


def inputs(args):
    table, subjects, names, evidence = pilot_table(args.pilots)
    rows, trees, observer, data, parent = data_source.inputs(SimpleNamespace(expanded_fit=args.expanded_fit, output=args.output))
    spec = io.read(args.pilots/PILOTS[0]/'registration.json')
    old = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()})
    pilot_rows, pilot_trees, *_ = late.validated_complete(old)
    if {r['id'] for r in pilot_rows} != set(table):
        raise ValueError('Changed original paid pilot cohort')
    for row in pilot_rows:
        if fixed.fingerprint(trees[row['id']]['']) != fixed.fingerprint(pilot_trees[row['id']]['']):
            raise ValueError('Cached root differs from paid pilot root')
    ids = sorted(table); index = {(r['id'], r['path']): i for i, r in enumerate(data['records'])}
    indices = [index[qid, ''] for qid in ids]
    x = {family: data['features'][family][indices].copy() for family in nested.FAMILIES}
    y = np.asarray([[table[qid][name]['correct'] for name in names] for qid in ids], int)
    cost = np.asarray([[table[qid][name]['compute'] for name in names] for qid in ids], float)
    observer_cost = np.asarray([observer.values_by_key[qid, ''][3]*observer._recipe['parameters']/1e9/3.09 for qid in ids])
    folds = np.asarray([fixed.refined.fold_for(qid) for qid in ids])
    evidence.update(parent['input_evidence'])
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]), 'input_evidence': evidence,
        'paths': {k: str(getattr(args, k).resolve()) for k in ('pilots', 'expanded_fit', 'output')},
        'questions': 64, 'subject_counts': dict(Counter(subjects.values())), 'protocols': names,
        'known_before_registration': 'All fixed pilot aggregate accuracies, costs and novelty',
        'features': list(nested.FAMILIES), 'recipes': list(RECIPES), 'penalty': PENALTY,
        'linear': {'C': .01, 'max_iter': 2000, 'class_weight': None},
        'forest': {'n_estimators': 96, 'max_depth': 4, 'min_samples_leaf': 8, 'random_state': 2381129},
        'prior_weight': .05, 'cost_prediction': 'Excluded-fold training mean for each fixed trajectory',
        'folds': 'Existing SHA256 question id modulo3', 'hyperparameter_search': False,
        'scope': 'Root routing feasibility only; no POMDP, controller or fresh result claim',
        'paid_visibility': 'Only original root features; no review/continuation text or answer before selection',
        'compute': 'All physical trajectory costs plus one paid root observer; CPU head latency separate',
        'screen': 'At least2pp above the highest fixed-protocol accuracy AND below retained rootstrong mean compute',
        'cloud_calls': 0, 'new_neural_calls': 0, 'development_or_test_read': False,
        'ids_sha256': fixed.fingerprint(ids), 'label_sha256': fixed.array_sha(y),
        'cost_sha256': fixed.array_sha(cost), 'observer_cost_sha256': fixed.array_sha(observer_cost),
        'feature_sha256': {k: fixed.array_sha(v) for k, v in x.items()}}
    return ids, names, x, y, cost, observer_cost, folds, record


def predict(x, y, cost, train, held, recipe):
    if np.any(train & held) or not train.any() or not held.any():
        raise ValueError('Nonoverlapping nonempty question folds required')
    probability = np.empty((int(held.sum()), y.shape[1]))
    for action in range(y.shape[1]):
        target = y[train, action]; prior = (target.sum()+1)/(len(target)+2)
        if len(np.unique(target)) == 1:
            p = np.full(int(held.sum()), prior)
        else:
            if recipe == 'linear':
                model = make_pipeline(StandardScaler(), LogisticRegression(C=.01, max_iter=2000))
            elif recipe == 'forest':
                model = ExtraTreesClassifier(n_estimators=96, max_depth=4, min_samples_leaf=8, random_state=2381129, n_jobs=1)
            else:
                raise ValueError('Unknown fixed recipe')
            model.fit(x[train], target); p = model.predict_proba(x[held])[:, list(model.classes_).index(1)]
        probability[:, action] = .95*p+.05*prior
    return probability, np.tile(cost[train].mean(0), (int(held.sum()), 1))


def diagnose(args, ids, names, x, y, cost, observer_cost, folds, record):
    controls = {name: {'n': 64, 'correct': int(y[:, j].sum()), 'accuracy': float(y[:, j].mean()),
        'mean_compute': float(cost[:, j].mean())} for j, name in enumerate(names)}
    best = max(controls, key=lambda name: (controls[name]['accuracy'], -controls[name]['mean_compute']))
    ceiling = controls['Strong retained second repair']['mean_compute']
    reports = {}; decisions = {}; receipts = []
    for family, features in x.items():
        for recipe in RECIPES:
            probability = np.full(y.shape, np.nan); prices = np.full(cost.shape, np.nan)
            for fold in range(3):
                train, held = folds != fold, folds == fold
                probability[held], prices[held] = predict(features, y, cost, train, held, recipe)
                receipts.append({'family': family, 'recipe': recipe, 'fold': fold,
                    'train_ids': [qid for qid, flag in zip(ids, train) if flag],
                    'held_ids': [qid for qid, flag in zip(ids, held) if flag]})
            if not np.isfinite(probability).all() or not np.isfinite(prices).all():
                raise ValueError('Incomplete held predictions')
            chosen = (probability-PENALTY*prices/1000).argmax(1)
            accuracy = y[np.arange(64), chosen]; paid = cost[np.arange(64), chosen]+observer_cost
            name = family+'-'+recipe
            reports[name] = {'n': 64, 'correct': int(accuracy.sum()), 'accuracy': float(accuracy.mean()),
                'mean_compute': float(paid.mean()), 'above_best_fixed': float(accuracy.mean()-controls[best]['accuracy']),
                'below_retained_rootstrong_compute': bool(paid.mean()<ceiling),
                'screen_passed': bool(accuracy.mean()>=controls[best]['accuracy']+.02 and paid.mean()<ceiling),
                'chosen_protocol_counts': dict(Counter(names[j] for j in chosen))}
            decisions[name] = [{'id': qid, 'protocol': names[a], 'correct': bool(ok), 'compute': float(c)}
                for qid, a, ok, c in zip(ids, chosen, accuracy, paid)]
    io.write(args.output/'private-decisions.json', decisions); io.write(args.output/'fold-receipts.json', receipts)
    result = {'schema': SCHEMA, 'fixed_controls': controls, 'highest_fixed_accuracy_control': best,
        'retained_rootstrong_compute_ceiling': ceiling, 'reports': reports,
        'hindsight_protocol_union': int(y.any(1).sum()), 'fresh_validation': False,
        'controller_promoted': False, 'development_or_test_read': False, 'cloud_calls': 0, 'new_neural_calls': 0}
    io.write(args.output/'summary.json', result); io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'any_screen_passed': any(v['screen_passed'] for v in reports.values()),
        'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})
    print({'best_fixed': controls[best], 'reports': reports}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=('register', 'diagnose'))
    for name in ('pilots', 'expanded-fit', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--registration-sha256'); args = parser.parse_args()
    with fixed.training_only_reads(), fixed.guarded.no_model_calls(), threadpool_limits(limits=1):
        bundle = inputs(args); record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record):
            raise ValueError('Changed registered portfolio diagnostic')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json', '.run.lock') for p in args.output.iterdir()):
                raise ValueError('Preserve completed or partial diagnostic')
            diagnose(args, *bundle)


if __name__ == '__main__': main()
