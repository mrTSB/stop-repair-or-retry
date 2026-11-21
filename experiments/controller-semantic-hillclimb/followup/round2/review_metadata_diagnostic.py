"""Strictly question-excluded CPU diagnostic of paid review metadata only."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import banks
from banks import committed
import review_metadata_spec as spec
import prepare_mixed_mentor as prepare
from prepare_mixed_mentor import sha, fingerprint
from context_pomdp import equal
from scoring import correct
import second_review_pilot as critical
import independent_review_pilot as independent


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def features(row, current, judge, protocol):
    """Only explicit public/paid fields enter this fixed feature vector."""
    if protocol not in ('root', 'critical', 'independent') or row['subject'] not in ('math', 'chemistry', 'physics'):
        raise ValueError('Unexpected registered review subject/protocol')
    calls = judge['calls']
    if not calls or len(calls) != judge['physical_api_calls']:
        raise ValueError('Incomplete physical review receipt')
    for call in calls:
        if (call['input_tokens'] <= 0 or call['output_cap'] <= 0
                or not 0 <= call['output_tokens'] <= call['output_cap']):
            raise ValueError('Invalid paid phase token receipt')
    if (sum(c['input_tokens'] for c in calls), sum(c['output_tokens'] for c in calls)) != (
            judge['input_tokens'], judge['output_tokens']):
        raise ValueError('Logical/physical review token totals disagree')
    thought = str(judge['verdict'].get('solution', ''))
    words = thought.split()
    p_current = judge.get('p_correct', .5)
    if type(p_current) not in (int, float) or not math.isfinite(p_current) or not 0 <= p_current <= 1:
        p_current = .5
    values = {name: float(row['subject'] == name.removeprefix('subject_')) for name in spec.FEATURES[:3]}
    values.update({name: float(protocol == name.removeprefix('protocol_')) for name in spec.FEATURES[3:6]})
    values.update({name: float(judge['action'] == name.removeprefix('action_')) for name in spec.FEATURES[6:9]})
    values.update(answer_present=float(bool(judge['answer'])), parse_valid=float(bool(judge['parse_valid'])),
        json_complete=float(bool(judge.get('json_complete', False))),
        agreement=float(bool(judge['answer'] and current['answer'] and equal(judge['answer'], current['answer']))),
        current_answer_present=float(bool(current['answer'])), p_current=float(p_current),
        log_phase_count=math.log1p(len(calls)), log_review_input_tokens=math.log1p(judge['input_tokens']),
        log_review_output_tokens=math.log1p(judge['output_tokens']),
        total_output_fraction=judge['output_tokens'] / sum(c['output_cap'] for c in calls),
        first_output_fraction=calls[0]['output_tokens'] / calls[0]['output_cap'],
        last_output_fraction=calls[-1]['output_tokens'] / calls[-1]['output_cap'],
        first_at_cap=float(calls[0]['output_tokens'] >= calls[0]['output_cap'] or calls[0].get('finish') == 'length'),
        last_at_cap=float(calls[-1]['output_tokens'] >= calls[-1]['output_cap'] or calls[-1].get('finish') == 'length'),
        log_thought_chars=math.log1p(len(thought)), log_thought_words=math.log1p(len(words)),
        log_check_words=math.log1p(len(str(judge['verdict'].get('check', '')).split())),
        log_current_words=math.log1p(len(current['text'].split())),
        log_answer_chars=math.log1p(len(judge['answer'])))
    for name, pattern in spec.MARKERS.items():
        values[name + '_markers_per100'] = 100. * len(re.findall(pattern, thought, flags=re.I)) / max(1, len(words))
    if set(values) != set(spec.FEATURES) or not all(math.isfinite(value) for value in values.values()):
        raise ValueError('Feature vector differs from its fixed specification')
    return np.asarray([values[name] for name in spec.FEATURES], dtype=float)


def fold(qid, inner=False):
    text = 'review-meta-inner-v1:' + qid if inner else qid
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 3


def weights(records):
    counts = Counter(row['id'] for row in records)
    return np.asarray([1. / counts[row['id']] for row in records])


def clipped(values):
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1. - 1e-6)


def calibrated(probabilities, temperature, records):
    p = clipped(probabilities)
    z = (np.log(p) - np.log1p(-p)) / temperature
    result = 1. / (1. + np.exp(-np.clip(z, -40, 40)))
    result[[not row['answer_present'] for row in records]] = 1e-6
    return clipped(result)


def binary_loss(y, probabilities, sample_weight):
    p = clipped(probabilities)
    return float(np.average(-np.asarray(y) * np.log(p) - (1. - np.asarray(y)) * np.log1p(-p), weights=sample_weight))


def train_model(name, records):
    y = np.asarray([row['target'] for row in records]); w = weights(records)
    prior = float((np.dot(w, y) + 1.) / (w.sum() + 2.))
    if name == 'subject_protocol_prior':
        table = {}
        for subject in ('math', 'chemistry', 'physics'):
            for protocol in ('root', 'critical', 'independent'):
                selected = np.asarray([row['subject'] == subject and row['protocol'] == protocol for row in records])
                table[subject, protocol] = float((np.dot(w[selected], y[selected]) + 2. * prior) / (w[selected].sum() + 2.))
        return {'name': name, 'prior': prior, 'table': table}
    chosen = spec.MODELS[name]['features']
    indexes = [spec.FEATURES.index(feature) for feature in chosen]
    if len(set(y)) < 2:
        return {'name': name, 'prior': prior, 'constant': True, 'indexes': indexes}
    x = np.asarray([row['features'] for row in records])[:, indexes]
    if name.endswith('logistic'):
        model = Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(C=1., max_iter=1000,
                            random_state=spec.SEED))])
        model.fit(x, y, scale__sample_weight=w, model__sample_weight=w)
    else:
        model = HistGradientBoostingClassifier(**{key: value for key, value in spec.MODELS[name].items()
                                                  if key != 'features'})
        model.fit(x, y, sample_weight=w)
    return {'name': name, 'model': model, 'indexes': indexes}


def predict(model, records):
    if model['name'] == 'subject_protocol_prior':
        p = [model['table'][row['subject'], row['protocol']] for row in records]
    elif model.get('constant'):
        p = np.full(len(records), model['prior'])
    else:
        x = np.asarray([row['features'] for row in records])[:, model['indexes']]
        classes = model['model'].classes_.astype(int)
        p = model['model'].predict_proba(x)[:, list(classes).index(1)]
    return calibrated(p, 1., records)


def outer_prediction(name, records, held_fold):
    training = [row for row in records if fold(row['id']) != held_fold]
    held = [row for row in records if fold(row['id']) == held_fold]
    fit_ids, held_ids = {row['id'] for row in training}, {row['id'] for row in held}
    if not fit_ids or not held_ids or fit_ids & held_ids:
        raise ValueError('Invalid whole-question outer split')
    receipt = {'outer_fold': held_fold, 'training_ids': sorted(fit_ids), 'held_ids': sorted(held_ids),
               'training_questions': len(fit_ids), 'held_questions': len(held_ids),
               'training_rows': len(training), 'held_rows': len(held), 'inner': []}
    temperature = 1.
    if name != 'subject_protocol_prior':
        probabilities = np.full(len(training), np.nan)
        for inner in range(3):
            inner_training = [row for row in training if fold(row['id'], True) != inner]
            indices = [i for i, row in enumerate(training) if fold(row['id'], True) == inner]
            inner_held = [training[i] for i in indices]
            train_ids, validation_ids = {r['id'] for r in inner_training}, {r['id'] for r in inner_held}
            if not train_ids or not validation_ids or train_ids & validation_ids or (train_ids | validation_ids) & held_ids:
                raise ValueError('Nested calibration included an outer held question')
            fitted = train_model(name, inner_training)
            probabilities[indices] = predict(fitted, inner_held)
            receipt['inner'].append({'fold': inner, 'training_ids': sorted(train_ids), 'held_ids': sorted(validation_ids),
                'model_sha256': hashlib.sha256(pickle.dumps(fitted, protocol=5)).hexdigest()})
        if not np.isfinite(probabilities).all():
            raise ValueError('Incomplete inner question-excluded predictions')
        y, w = [r['target'] for r in training], weights(training)
        search = [{'temperature': t, 'loss': binary_loss(y, calibrated(probabilities, t, training), w)}
                  for t in spec.TEMPERATURES]
        temperature = min(search, key=lambda row: (row['loss'], row['temperature']))['temperature']
        receipt['temperature_search'] = search
    fitted = train_model(name, training)
    result = calibrated(predict(fitted, held), temperature, held)
    receipt['temperature'] = temperature
    return held, result, fitted, receipt


def metrics(records, probabilities):
    y, w = np.asarray([r['target'] for r in records]), weights(records)
    p = clipped(probabilities)
    order = sorted(range(len(records)), key=lambda i: (-p[i], records[i]['id'], records[i]['protocol']))
    coverage = {}
    for fraction in (.25, .5, .75, 1.):
        budget = float(w.sum() * fraction); remaining = budget; correct_mass = 0.
        for index in order:
            take = min(remaining, float(w[index])); correct_mass += take * y[index]; remaining -= take
            if remaining <= 1e-12:
                break
        coverage[str(fraction)] = float(correct_mass / budget)
    return {'questions': len({r['id'] for r in records}), 'records': len(records),
            'correct_fraction': float(np.average(y, weights=w)),
            'brier': float(np.average((p - y) ** 2, weights=w)),
            'binary_log_loss': binary_loss(y, p, w),
            'auroc': float(roc_auc_score(y, p, sample_weight=w)) if len(set(y)) == 2 else None,
            'accuracy_at_half': float(np.average((p >= .5) == y, weights=w)),
            'coverage_precision': coverage}


def panel(record, name):
    return {'new_root': record['protocol'] == 'root' and record['bank'] == 'new',
        'new_science_root': record['protocol'] == 'root' and record['bank'] == 'new' and record['subject'] != 'math',
        'all_root': record['protocol'] == 'root', 'old_root': record['protocol'] == 'root' and record['bank'] == 'old',
        'all_reviews': True, 'root_math': record['protocol'] == 'root' and record['subject'] == 'math',
        'root_chemistry': record['protocol'] == 'root' and record['subject'] == 'chemistry',
        'root_physics': record['protocol'] == 'root' and record['subject'] == 'physics',
        'critical': record['protocol'] == 'critical', 'independent': record['protocol'] == 'independent'}[name]


def comparisons(records, predictions):
    groups = defaultdict(list)
    for index, row in enumerate(records):
        groups[row['id']].append(index)
    qids = sorted(groups); y = np.asarray([r['target'] for r in records])
    losses = {}
    for name, values in predictions.items():
        p = clipped(values)
        losses[name] = {'brier': (p - y) ** 2,
                       'binary_log_loss': -y * np.log(p) - (1. - y) * np.log1p(-p)}
    draws = np.random.default_rng(spec.SEED).integers(0, len(qids), size=(2000, len(qids)))
    result = {}
    for name in predictions:
        for baseline in ('subject_protocol_prior', 'confidence_agreement_logistic'):
            if name == baseline:
                continue
            result[name + '_minus_' + baseline] = {}
            for metric in ('brier', 'binary_log_loss'):
                differences = losses[name][metric] - losses[baseline][metric]
                per_question = np.asarray([differences[groups[qid]].mean() for qid in qids])
                result[name + '_minus_' + baseline][metric] = {'point': float(per_question.mean()),
                    'ci95': np.quantile(per_question[draws].mean(1), [.025, .975]).tolist(),
                    'lower_is_better': True}
    return result


def load_records(args, registration):
    source = {name: Path(path) for name, path in registration['source_directories'].items()}
    for name, path in source.items():
        if sha(path / 'contract.json') != registration['source_contract_sha256'][name]:
            raise ValueError('Registered review protocol changed')
    if (sha(source['root'] / 'complete.json'), sha(source['critical'] / 'summary.json')) != (
            registration['root_complete_sha256'], registration['critical_summary_sha256']):
        raise ValueError('Completed root/critical controls changed')
    done = read(source['independent'] / 'summary.json')
    if (done['questions'], done['test_used'], done['bank_sha256'], done['contract_sha256'],
            done['physical_phases_sha256']) != (64, False, sha(source['independent'] / 'train.jsonl'),
            sha(source['independent'] / 'contract.json'), sha(source['independent'] / 'physical-phases.jsonl')):
        raise ValueError('Finish all64 independent pairs and preserve their full receipts')
    view_args = argparse.Namespace(old_data=args.old_data, new_data=args.new_data, mentor=source['root'])
    rows, trees, judges, _, contract = prepare.training_view(view_args)
    parent_identity = read(args.parent_models / 'identity.json')
    parent_complete = read(args.parent_models / 'complete.json')
    if parent_complete['identity_sha256'] != sha(args.parent_models / 'identity.json'):
        raise ValueError('Parent training identity differs from its completion receipt')
    if fingerprint(trees) != parent_identity['tree_sha256']:
        raise ValueError('Root training tree differs from its completed parent fit')
    parent_neural = read(args.parent_models / 'neural-receipt.json')
    if parent_neural['metadata_sha256'] != sha(args.parent_models / 'metadata.jsonl'):
        raise ValueError('Archived root training target metadata changed')
    parent_targets = {r['id']: r['label'] % 2 for r in (
        json.loads(line) for line in (args.parent_models / 'metadata.jsonl').read_text().splitlines())
        if r['kind'] == 'original-review' and r['path'] == ''}
    by_id = {r['id']: r for r in rows}; chosen = {r['id']: r for r in contract['selection']}
    control_args = argparse.Namespace(data=args.new_data, mentor=source['root'], critical=source['critical'])
    _, science_rows, _, strong, earlier, _ = independent.controls(control_args)
    raw = committed(source['independent'] / 'train.jsonl')
    known = {(r['id'], r['kind']): r['response'] for r in raw}
    expected = {(r['id'], kind) for r in science_rows for kind in ('judge', 'solver')}
    if len(raw) != 128 or set(known) != expected:
        raise ValueError('Need every independent review/repair sibling exactly once')
    phases = committed(source['independent'] / 'physical-phases.jsonl')
    phase_lookup = {(r['id'], r['phase']): r['response'] for r in phases}
    if len(phases) != 128 or len(phase_lookup) != 128:
        raise ValueError('Incomplete independent physical phase grid')
    records = []

    def add(row, current, judge, protocol):
        safe = {**row, 'answer': ''}
        values = features(safe, current, judge, protocol)
        target = int(correct(judge, row['answer']))
        if protocol == 'root' and target != parent_targets[row['id']]:
            raise ValueError('Root correctness target differs from the archived training reference')
        if target and not judge['answer']:
            raise ValueError('An invalid answer cannot be correct')
        records.append({'id': row['id'], 'subject': row['subject'], 'bank': chosen[row['id']]['bank'],
            'protocol': protocol, 'features': values.tolist(), 'target': target,
            'answer_present': bool(judge['answer']), 'review_sha256': fingerprint(judge),
            'current_sha256': fingerprint(current)})

    for qid in chosen:
        add(by_id[qid], trees[qid][''], judges[qid, ''], 'root')
    for row in science_rows:
        qid = row['id']; safe = {**row, 'answer': ''}; current = strong[qid, 'solver', 'guided']
        teacher = known[qid, 'judge']
        independent.verify_review(safe, current, teacher)
        independent.verify_solver(safe, current, teacher, known[qid, 'solver'])
        if any(phase_lookup.get((qid, call['phase'])) != call for call in teacher['calls']):
            raise ValueError('Independent logical/physical review receipts disagree')
        add(row, current, earlier[qid, 'judge'], 'critical')
        add(row, current, teacher, 'independent')
    if len(records) != 320 or len({r['id'] for r in records}) != 192:
        raise ValueError('Diagnostic requires all320 records on192 questions')
    root = Path(__file__).resolve().parents[4]
    source_files = {str(Path(module.__file__).resolve()): sha(module.__file__)
                    for module in list(sys.modules.values()) if getattr(module, '__file__', None)
                    and Path(module.__file__).is_file() and Path(module.__file__).resolve().is_relative_to(root)}
    evidence = {str(path.resolve()): sha(path) for path in [
        args.old_data / 'splits/train-questions.jsonl', args.new_data / 'splits/train-questions.jsonl',
        args.parent_models / 'identity.json', args.parent_models / 'complete.json',
        args.parent_models / 'neural-receipt.json', args.parent_models / 'metadata.jsonl', source['root'] / 'train.jsonl',
        source['critical'] / 'train.jsonl', source['independent'] / 'train.jsonl',
        source['independent'] / 'physical-phases.jsonl', source['independent'] / 'summary.json']}
    return records, {'source_files': source_files, 'input_evidence': evidence,
                     'records_sha256': fingerprint(records)}


def run(args):
    registration_path = args.output / 'registration.json'
    registration = read(registration_path)
    if sha(spec.__file__) != registration['specification_sha256']:
        raise ValueError('Prespecified diagnostic features/models changed')
    if (registration['features'], registration['models'], registration['temperature_grid']) != (
            spec.FEATURES, spec.MODELS, spec.TEMPERATURES):
        raise ValueError('Implementation differs from preregistration')
    if (args.output / 'implementation.json').exists() or (args.output / 'summary.json').exists():
        raise ValueError('Never overwrite a completed or previously begun diagnostic')
    waiting = Path(registration['source_directories']['independent']) / 'summary.json'
    while not waiting.exists():
        if not args.wait:
            raise ValueError('Complete every independent pair before full diagnostic scoring; use --wait')
        time.sleep(30)
    with threadpool_limits(limits=1):
        records, provenance = load_records(args, registration)
        write(args.output / 'implementation.json', {**provenance,
            'registration_sha256': sha(registration_path), 'neural_calls': 0, 'cloud_calls': 0,
            'project_dev_test_values_read': False, 'nested_calibration': True})
        (args.output / 'private-features.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        positions = {(r['id'], r['protocol']): i for i, r in enumerate(records)}
        predictions, receipts = {}, {}
        for name in spec.MODELS:
            values = np.full(len(records), np.nan); receipts[name] = []
            for held_fold in range(3):
                held, probability, fitted, receipt = outer_prediction(name, records, held_fold)
                for row, value in zip(held, probability):
                    values[positions[row['id'], row['protocol']]] = value
                model_path = args.output / f'{name}-outer{held_fold}.pkl'
                model_path.write_bytes(pickle.dumps(fitted, protocol=5))
                receipt['model_sha256'] = sha(model_path); receipts[name].append(receipt)
            if not np.isfinite(values).all():
                raise ValueError('Incomplete held-question predictions')
            predictions[name] = values
        np.savez_compressed(args.output / 'private-predictions.npz', **predictions)
        write(args.output / 'fold-receipts.json', receipts)
        reports = {}
        for name in spec.REPORTS:
            indices = [i for i, row in enumerate(records) if panel(row, name)]
            selected = [records[i] for i in indices]
            reports[name] = {model: metrics(selected, values[indices]) for model, values in predictions.items()}
            if name in registration['primary_panels']:
                reports[name]['paired_comparisons'] = comparisons(selected, {k: v[indices] for k, v in predictions.items()})
        for path, digest in {**provenance['source_files'], **provenance['input_evidence']}.items():
            if sha(path) != digest:
                raise ValueError('Diagnostic evidence/source changed while fitting')
        result = {'registration_sha256': sha(registration_path),
            'implementation_sha256': sha(args.output / 'implementation.json'),
            'private_features_sha256': sha(args.output / 'private-features.jsonl'),
            'predictions_sha256': sha(args.output / 'private-predictions.npz'),
            'fold_receipts_sha256': sha(args.output / 'fold-receipts.json'),
            'questions': 192, 'records': 320, 'reports': reports,
            'question_weights': 'Total1 per question within each training/report panel',
            'strict_nested_question_calibration': True, 'neural_calls': 0, 'cloud_calls': 0,
            'runtime_policy_changed': False, 'development_or_test_used': False,
            'interpretation': 'Training-only grouped diagnostic; all prespecified families reported; not a fresh-data C result.'}
        write(args.output / 'summary.json', result)
        print(json.dumps({'complete': True, 'questions': 192, 'records': 320,
            'primary': {name: reports[name] for name in registration['primary_panels']}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-data', 'new-data', 'parent-models', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    run(parser.parse_args())
