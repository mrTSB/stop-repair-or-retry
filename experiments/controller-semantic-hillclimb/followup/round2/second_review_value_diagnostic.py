"""Training-only value of an independent review after the first guided answer.

All64 already registered science questions are retained. A fixed, question-
excluded forecaster estimates the joint correctness of stopping, retaining the
old review, or buying the independent review. It sees only the paid prefix.
This is an exploratory feasibility diagnostic, not a selected controller or a
fresh test. Earlier aggregate action outcomes were known before registration.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import independent_review_pilot as pilot
import paid_review_metadata as paid
import review_metadata_diagnostic as metadata
import review_metadata_spec as spec
from banks import committed
from context_pomdp import equal
from prepare_mixed_mentor import fingerprint, sha
from scoring import correct

FEATURES = spec.FEATURES + ['guided_answer_present', 'guided_agrees_with_review',
    'guided_agrees_with_root', 'log_guided_words', 'log_guided_input_tokens',
    'guided_output_fraction', 'guided_at_cap']
PENALTIES = [0., .02, .04, .08, .16]
ACTIONS = ['stop_after_guided', 'retained_repair', 'independent_repair']
REWARDS = np.asarray([[(state >> bit) & 1 for bit in range(3)] for state in range(8)])


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def public_features(row, root, review, guided):
    entry = paid.public_entry(row, root, review)
    base = metadata.features(entry['row'], entry['current'], entry['judge'], 'root')
    extra = [float(bool(guided['answer'])),
        float(bool(guided['answer'] and review['answer'] and equal(guided['answer'], review['answer']))),
        float(bool(guided['answer'] and root['answer'] and equal(guided['answer'], root['answer']))),
        np.log1p(len(guided['text'].split())), np.log1p(guided['input_tokens']),
        guided['output_tokens'] / guided['output_cap'],
        float(guided['output_tokens'] >= guided['output_cap'] or guided.get('finish') == 'length')]
    values = np.r_[base, extra]
    if len(values) != len(FEATURES) or not np.isfinite(values).all():
        raise ValueError('Invalid paid-prefix features')
    return values


def fold(qid):
    return int(hashlib.sha256(('second-review-value-v1:' + qid).encode()).hexdigest()[:8], 16) % 3


def cost(node, billions):
    return (node['input_tokens'] + node['output_tokens']) * billions / 3.09


def source_files():
    project = Path(__file__).resolve().parents[4]
    here = Path(__file__).resolve().parent
    def allowed(path):
        if path is None:
            return None
        path = Path(path).resolve()
        if (path.is_file() and path.suffix == '.py' and path.is_relative_to(project)
                and not any(part.startswith('.') or part in ('site-packages', '__pycache__')
                            for part in path.relative_to(project).parts)):
            return path
        return None
    def locate(name):
        if not name:
            return None
        if name in sys.modules:
            return allowed(getattr(sys.modules[name], '__file__', None))
        relative = Path(*name.split('.'))
        for folder in dict.fromkeys([str(here), *sys.path]):
            for path in (Path(folder or '.') / relative.with_suffix('.py'), Path(folder or '.') / relative / '__init__.py'):
                if path.is_file():
                    return allowed(path)
        return None
    found, pending = set(), [(Path(__file__).resolve(), 'second_review_value_diagnostic')]
    while pending:
        path, module = pending.pop()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                package = module.split('.')[:-node.level] if node.level else []
                name = '.'.join([*package, *([node.module] if node.module else [])])
                names = [name, *[name + '.' + alias.name for alias in node.names if alias.name != '*']]
            for name in names:
                dependency = locate(name)
                if dependency is not None:
                    pending.append((dependency, name))
    return {str(path): sha(path) for path in sorted(found)}


def verify_physical(rows, independent, raw_phases):
    phases = {(r['id'], r['phase']): r['response'] for r in raw_phases}
    expected = {(r['id'], phase) for r in rows for phase in (pilot.THOUGHT_PHASE, pilot.ANSWER_PHASE)}
    if len(phases) != len(raw_phases) or set(phases) != expected:
        raise ValueError('Incomplete, duplicate or extra independent physical phases')
    for row in rows:
        calls = independent[row['id'], 'judge']['calls']
        if [c['phase'] for c in calls] != [pilot.THOUGHT_PHASE, pilot.ANSWER_PHASE]:
            raise ValueError('Changed physical phase order')
        if any(phases[row['id'], call['phase']] != call for call in calls):
            raise ValueError('Logical review differs from physical phase ledger')


def registration(args):
    complete = read(args.independent / 'summary.json')
    if (complete['questions'], complete['test_used'], complete['bank_sha256'], complete['contract_sha256']) != (
            64, False, sha(args.independent / 'train.jsonl'), sha(args.independent / 'contract.json')):
        raise ValueError('Require the complete unchanged independent training pilot')
    if complete['physical_phases_sha256'] != sha(args.independent / 'physical-phases.jsonl'):
        raise ValueError('Independent physical phase ledger changed after completion')
    return {'schema': 'second-review-value-diagnostic-v1', 'source_files': source_files(),
        'inputs': {name: str(getattr(args, name).resolve()) for name in ('data', 'mentor', 'critical', 'independent')},
        'input_hashes': {str(path.resolve()): sha(path) for path in [
            args.data / 'splits/train-questions.jsonl', args.mentor / 'contract.json',
            args.mentor / 'complete.json', args.mentor / 'train.jsonl',
            args.critical / 'contract.json', args.critical / 'summary.json', args.critical / 'train.jsonl',
            args.critical / 'prefix-receipt.json', args.independent / 'contract.json',
            args.independent / 'summary.json', args.independent / 'train.jsonl', args.independent / 'physical-phases.jsonl']},
        'features': FEATURES, 'questions': 64, 'subjects': {'chemistry': 32, 'physics': 32},
        'classifier': {'family': 'standardized multinomial logistic', 'C': .1, 'temperature': 1., 'max_iter': 2000},
        'future_cost_model': {'family': 'standardized ridge on log1p cost', 'alpha': 10.,
            'prediction_clip': 'zero to twice the maximum training log1p cost'},
        'states': 'Three correctness bits for stop, retained repair, independent repair',
        'folds': 'SHA256(second-review-value-v1:question_id) first8hex modulo3',
        'penalties': PENALTIES, 'action_order_for_exact_ties': ACTIONS,
        'criterion': 'Expected correctness minus penalty*predicted incremental compute/1000',
        'known_before_registration': 'All pilot aggregate action accuracies and novelty; exploratory training diagnostic',
        'reporting': 'Every fixed penalty, every question; no selected threshold or deployment claim',
        'all_paid_prefix_and_successful_physical_phases_charged': True,
        'no_second_review_features_before_purchase': True,
        'development_or_test_used': False, 'neural_calls': 0, 'cloud_calls': 0}


def load_records(args):
    _, rows, trees, strong, _, hashes = pilot.controls(args)
    raw = committed(args.independent / 'train.jsonl')
    independent = {(r['id'], r['kind']): r['response'] for r in raw}
    if len(raw) != 128 or set(independent) != {(r['id'], kind) for r in rows for kind in ('judge', 'solver')}:
        raise ValueError('Require every independent judge/solver pair without duplicates')
    verify_physical(rows, independent, committed(args.independent / 'physical-phases.jsonl'))
    result = []
    for row in rows:
        qid = row['id']; safe = {**row, 'answer': ''}
        root = trees[qid]['']; review = strong[qid, 'judge', '']; guided = strong[qid, 'solver', 'guided']
        retained = strong[qid, 'solver', 'guided/reuse']
        second, last = independent[qid, 'judge'], independent[qid, 'solver']
        pilot.verify_review(safe, guided, second); pilot.verify_solver(safe, guided, second, last)
        labels = [int(correct(node, row['answer'])) for node in (guided, retained, last)]
        result.append({'id': qid, 'subject': row['subject'], 'fold': fold(qid),
            'features': public_features(safe, root, review, guided).tolist(),
            'labels': labels, 'state': sum(y << bit for bit, y in enumerate(labels)),
            'prefix_compute': cost(root, 4.) + cost(review, 3.) + cost(guided, 4.),
            'future_compute': [0., cost(retained, 4.), cost(second, 3.) + cost(last, 4.)],
            'paid_prefix_sha256': fingerprint([root, review, guided])})
    if len(result) != 64 or len({r['id'] for r in result}) != 64:
        raise ValueError('Use all64 unique registered science questions')
    return result, fingerprint(hashes)


def predict_excluded(records):
    x = np.asarray([r['features'] for r in records]); y = np.asarray([r['state'] for r in records])
    groups = np.asarray([r['fold'] for r in records])
    probabilities = np.zeros((len(records), 8)); future = np.zeros((len(records), 3)); receipts = []
    for held in range(3):
        train, test = groups != held, groups == held
        if not train.any() or not test.any():
            raise ValueError('Every held question fold must be represented')
        if len(set(y[train])) == 1:
            probabilities[test, int(y[train][0])] = 1.
        else:
            classifier = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=2000, random_state=20261123))
            classifier.fit(x[train], y[train])
            probabilities[np.ix_(np.flatnonzero(test), classifier[-1].classes_.astype(int))] = classifier.predict_proba(x[test])
        # Held future costs are never supplied to the predictor or decision rule.
        target = np.log1p(np.asarray([r['future_compute'][1:] for r in records])[train])
        predictor = make_pipeline(StandardScaler(), Ridge(alpha=10.))
        predictor.fit(x[train], target)
        future[test, 1:] = np.expm1(np.clip(predictor.predict(x[test]), 0., 2. * target.max(axis=0)))
        receipts.append({'fold': held, 'training_ids': [r['id'] for r in records if r['fold'] != held],
            'excluded_ids': [r['id'] for r in records if r['fold'] == held]})
    if not np.isfinite(probabilities).all() or not np.isfinite(future).all():
        raise ValueError('Nonfinite question-excluded forecasts')
    np.testing.assert_allclose(probabilities.sum(1), 1.)
    return probabilities, future, receipts


def decisions(probabilities, predicted_costs, penalty):
    return np.argmax(probabilities @ REWARDS - penalty * predicted_costs / 1000., axis=1)


def aggregate(records, choices):
    values = [dict(correct=row['labels'][int(choice)],
        compute=row['prefix_compute'] + row['future_compute'][int(choice)]) for row, choice in zip(records, choices)]
    return {'n': len(values), 'correct': sum(v['correct'] for v in values),
        'accuracy': sum(v['correct'] for v in values) / len(values),
        'mean_compute': float(np.mean([v['compute'] for v in values])),
        'actions': {name: int(np.sum(np.asarray(choices) == index)) for index, name in enumerate(ACTIONS)}}


def run(args):
    expected = registration(args)
    if args.mode == 'register':
        if args.output.exists():
            raise ValueError('Use a new registration directory')
        args.output.mkdir(parents=True); write(args.output / 'registration.json', expected)
        print(json.dumps({'registered': True, 'registration_sha256': sha(args.output / 'registration.json'), 'scored': False}))
        return
    if read(args.output / 'registration.json') != expected or (args.output / 'summary.json').exists():
        raise ValueError('Changed registration or previously completed diagnostic')
    started = time.perf_counter()
    records, control_hash = load_records(args)
    with threadpool_limits(limits=1):
        probabilities, future, folds = predict_excluded(records)
    fixed = {name: aggregate(records, np.full(len(records), index)) for index, name in enumerate(ACTIONS)}
    variants = []
    private = []
    for penalty in PENALTIES:
        choices = decisions(probabilities, future, penalty)
        variants.append({'penalty': penalty, **aggregate(records, choices),
            'by_subject': {subject: aggregate([r for r in records if r['subject'] == subject],
                [a for r, a in zip(records, choices) if r['subject'] == subject]) for subject in ('chemistry', 'physics')}})
        private.append({'penalty': penalty, 'choices': choices.tolist()})
    write(args.output / 'private-predictions.json', {'records': records, 'folds': folds,
        'probabilities': probabilities.tolist(), 'predicted_future_compute': future.tolist(), 'variants': private})
    summary = {'n': len(records), 'fixed': fixed, 'variants': variants,
        'joint_action_oracle_correct': sum(max(r['labels']) for r in records),
        'registration_sha256': sha(args.output / 'registration.json'),
        'predictions_sha256': sha(args.output / 'private-predictions.json'), 'controls_sha256': control_hash,
        'elapsed_cpu_work_seconds': time.perf_counter() - started,
        'development_or_test_used': False, 'neural_calls': 0, 'cloud_calls': 0,
        'interpretation': 'Question-excluded exploratory training feasibility only; no selected controller or fresh result.'}
    write(args.output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'mentor', 'critical', 'independent', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mode', choices=['register', 'run'], required=True)
    run(parser.parse_args())
