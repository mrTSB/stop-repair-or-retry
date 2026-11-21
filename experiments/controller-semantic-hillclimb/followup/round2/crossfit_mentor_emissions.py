"""Rebuild retained-mentor emissions with question-excluded calibration.

The fitted runtime calibrator, beliefs, transitions, review model, solver
protocol and costs stay identical. Only first-action observation emissions
use calibration predictions that excluded the whole question. Inputs are the
already computed local neural outputs; no neural or cloud inference is needed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare_mixed_mentor import training_view, fingerprint, sha
import mentor_pomdp as mentor


def fit_calibrator(x, y):
    return LogisticRegression(C=1., max_iter=1000).fit(x, y)


def calibration_crossfit(features, labels, qids, train_mask):
    """Use the parent's valid solver training mask; hold out every question row."""
    x, y = np.asarray(features), np.asarray(labels)
    qids, train_mask = np.asarray(qids), np.asarray(train_mask, dtype=bool)
    folds = np.asarray([int(hashlib.sha256(q.encode()).hexdigest()[:8], 16) % 3 for q in qids])
    out = np.zeros((len(x), 4)); receipts = []
    for fold in (0, 1, 2):
        fit_mask, held = train_mask & (folds != fold), folds == fold
        fit_ids, held_ids = set(qids[fit_mask]), set(qids[held])
        if fit_ids & held_ids or not fit_ids or not held_ids:
            raise ValueError('Need three disjoint question groups for calibration')
        model = fit_calibrator(x[fit_mask], y[fit_mask])
        out[np.ix_(held, model.classes_.astype(int))] = model.predict_proba(x[held])
        receipts.append({'fold': fold, 'fit_questions': len(fit_ids), 'held_questions': len(held_ids),
                         'fit_rows': int(fit_mask.sum()), 'held_rows': int(held.sum()), 'question_overlap': 0,
                         'fit_ids_sha256': fingerprint(sorted(fit_ids)),
                         'held_ids_sha256': fingerprint(sorted(held_ids))})
    if not np.allclose(out.sum(1), 1.) or not np.isfinite(out).all():
        raise ValueError('Invalid question-excluded calibration probabilities')
    return out, receipts


def observation_support():
    """Exactly the parent's soft structural support, including its 1e-4 mass."""
    mask = np.ones((8, mentor.N_OBS * 3))
    for state in range(8):
        current, teacher = (state // 2) % 2, state % 2
        for observation in range(mentor.N_OBS):
            for agreement in range(3):
                if ((current and observation == 0) or (observation == 0 and agreement != 0)
                        or (agreement == 2 and current != teacher)
                        or (current == teacher == 1 and agreement != 2)):
                    mask[state, observation * 3 + agreement] = 1e-4
    return mask


def rebuild_emissions(records, subjects, smoothing):
    """Records are (subject, action, hidden_next_state, OOF_observation)."""
    support = observation_support()

    def counts(subject=None):
        values = np.broadcast_to(.1 * support, (2, 8, mentor.N_OBS * 3)).copy()
        for sub, action, state, observation in records:
            if subject is None or sub == subject:
                values[action, state, observation] += 1
        return values

    pooled = counts()
    pooled /= pooled.sum(-1, keepdims=True)
    result = {}
    for subject in subjects:
        values = counts(subject) + smoothing * pooled
        result[subject] = values / values.sum(-1, keepdims=True)
    return result


def preserved_digest(model):
    dynamics = {}
    for subject, d in model.mentor.dynamics.items():
        dynamics[subject] = {name: value for name, value in vars(d).items()
                             if name not in ('emission', 'mentor', '_alphas')}
    # The expected changed fields are emissions, their agreement marginal and
    # provenance receipts. Runtime calibration and all belief/transition/cost
    # machinery must retain identical serialized content.
    value = (model.base, model.transfer, model.mentor.regular,
             model.mentor.classifier, model.mentor.review_observer, dynamics)
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def run(args):
    while not (args.models / 'complete.json').exists():
        if not args.wait:
            raise ValueError('Finish the registered mixed-mentor fit or use --wait')
        time.sleep(30)
    original = json.loads((args.models / 'complete.json').read_bytes())
    original_identity = json.loads((args.models / 'identity.json').read_bytes())
    rows, trees, judges, old_ids, contract = training_view(args)
    if fingerprint(trees) != original_identity['tree_sha256']:
        raise ValueError('Original mentor model used another training tree')
    metadata_path, neural_path = args.models / 'metadata.jsonl', args.models / 'neural.npz'
    neural_receipt = json.loads((args.models / 'neural-receipt.json').read_bytes())
    if (sha(metadata_path), sha(neural_path)) != (neural_receipt['metadata_sha256'], neural_receipt['outputs_sha256']):
        raise ValueError('Original local neural outputs changed')
    metadata = [json.loads(s) for s in metadata_path.read_text().splitlines()]
    outputs = np.load(neural_path)
    if len(metadata) != len(outputs['state']):
        raise ValueError('Misaligned neural metadata')
    specs = original['models']
    first_path = args.models / (specs[0]['key'] + '.pkl')
    if sha(first_path) != specs[0]['sha256']:
        raise ValueError('Changed original model')
    first = pickle.loads(first_path.read_bytes())
    by_id = {r['id']: r for r in rows}
    solver_indexes = [i for i, r in enumerate(metadata) if r['kind'] == 'solver']
    solver_meta = [metadata[i] for i in solver_indexes]
    features = [first.mentor.classifier.calibration_features(by_id[r['id']], r['path'], outputs['state'][i])
                for i, r in zip(solver_indexes, solver_meta)]
    predictions, fold_receipts = calibration_crossfit(features, [r['label'] for r in solver_meta],
                                                      [r['id'] for r in solver_meta],
                                                      [r['train'] and r['valid'] for r in solver_meta])
    lookup = {(r['id'], r['path']): (r, predictions[i]) for i, r in enumerate(solver_meta)}
    if len(lookup) != len(solver_meta):
        raise ValueError('Duplicate solver calibration row')
    records = []
    paired = [r for r in rows if (r['id'], '') in judges]
    if len(paired) != 192:
        raise ValueError('Use all 192 registered mentor pairs')
    for row in paired:
        qid = row['id']; judge = judges[qid, '']; tree = trees[qid]
        teacher = int(mentor.correct(judge, row['answer']))
        hard = int(np.mean([mentor.correct(tree[p], row['answer']) for p in mentor.core.FIXED]) < .6)
        for action, path in enumerate(('guided', 'retry')):
            metadata_row, probability = lookup[qid, path]
            base_state = 2 * hard + int(mentor.correct(tree[path], row['answer']))
            if metadata_row['label'] != base_state or metadata_row['prompt_sha256'] != tree[path]['prompt_sha256']:
                raise ValueError('Calibration label or paid response metadata changed')
            obs = mentor.core.observation(probability, bool(tree[path]['answer']))
            records.append((row['subject'], action, mentor.state(base_state, teacher),
                            obs * 3 + mentor.comparison(judge, tree[path])))
    identity = {'source_sha256': sha(__file__), 'parent_mentor_source_sha256': sha(mentor.__file__),
                'source_complete_sha256': sha(args.models / 'complete.json'),
                'source_identity_sha256': sha(args.models / 'identity.json'),
                'neural_metadata_sha256': sha(metadata_path), 'neural_outputs_sha256': sha(neural_path),
                'neural_receipt_sha256': sha(args.models / 'neural-receipt.json'),
                'mentor_bank_sha256': sha(args.mentor / 'train.jsonl'),
                'new_train_question_sha256': sha(args.new_data / 'splits/train-questions.jsonl'),
                'tree_sha256': fingerprint(trees), 'fit_questions': len(rows),
                'calibration': 'Same valid solver train mask and C1 logistic regression; whole-question 3fold exclusion',
                'folds': fold_receipts, 'emission_records': len(records),
                'changes': ['mentor first-action emissions', 'derived mentor-comparison marginals'],
                'runtime_calibration_changed': False, 'transitions_costs_transfer_changed': False,
                'neural_calls': 0, 'cloud_calls': 0, 'development_or_test_used': False}
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_bytes()) != identity:
        raise ValueError('Changed emission refinement inputs; use a new output directory')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    np.savez_compressed(args.output / 'calibration-oof.npz', probabilities=predictions)
    models = []
    for spec in specs:
        source_path = args.models / (spec['key'] + '.pkl')
        if sha(source_path) != spec['sha256']:
            raise ValueError('Changed source hybrid model')
        model = pickle.loads(source_path.read_bytes())
        a, b = first.mentor.classifier, model.mentor.classifier
        if (a.temperature != b.temperature or not np.array_equal(a.calibration.classes_, b.calibration.classes_)
                or not np.array_equal(a.calibration.coef_, b.calibration.coef_)
                or not np.array_equal(a.calibration.intercept_, b.calibration.intercept_)):
            raise ValueError('Candidates used different original calibrations')
        before = preserved_digest(model)
        emissions = rebuild_emissions(records, model.mentor.dynamics, spec['smoothing'])
        for subject, d in model.mentor.dynamics.items():
            d.emission = emissions[subject]
            d.mentor = d.emission.reshape(2, 8, mentor.N_OBS, 3).sum(2)
        if preserved_digest(model) != before:
            raise ValueError('A non-emission component changed')
        refinement = {'parent_model_sha256': spec['sha256'], 'identity_sha256': sha(identity_path),
                      'question_excluded_calibration_emissions': True, 'calibration_folds': 3}
        model.mentor.fit_receipt.update(refinement)
        model.fit_receipt['mentor'] = model.mentor.fit_receipt
        model.fit_receipt['emission_refinement'] = refinement
        path = args.output / (spec['key'] + '.pkl')
        path.write_bytes(pickle.dumps(model, protocol=5))
        models.append({**spec, 'sha256': sha(path), 'parent_sha256': spec['sha256'],
                       'preserved_non_emission_sha256': before})
        print('Rebuilt question-excluded mentor emissions', spec['key'], flush=True)
    summary = {**original, 'models': models, 'identity_sha256': sha(identity_path),
               'parent_complete_sha256': sha(args.models / 'complete.json'),
               'development_or_test_used': False, 'neural_calls': 0, 'cloud_calls': 0}
    (args.output / 'complete.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-data', 'new-data', 'mentor', 'models', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    run(parser.parse_args())
