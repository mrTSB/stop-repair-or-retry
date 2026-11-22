"""Fit two science review-observation POMDPs from sealed TRAIN-only critics."""
import argparse
import copy
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import critic_review_hybrid as runtime
import expanded_mentor_core as core
import science_review_critic as diagnostic

io = diagnostic.io
SCHEMA = 'science-review-critic-POMDP-fit-v1'


def inputs(args):
    spec = io.read(args.diagnostic/'registration.json'); done = io.read(args.diagnostic/'complete.json')
    parent_args = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()},
                                 registration_sha256=io.sha(args.diagnostic/'registration.json'))
    rows, _, _, _, expected = diagnostic.load_inputs(parent_args)
    if expected != spec or done['registration_sha256'] != parent_args.registration_sha256:
        raise ValueError('Critic diagnostic changed')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    io.verify({str(args.diagnostic/n): h for n, h in done['digests'].items()})
    saved = diagnostic.restore(parent_args, rows, True)
    report = io.read(args.diagnostic/'graded/summary.json')
    if not all(report['reports'][n]['screen_passed'] for n in ('option', 'combined')):
        raise ValueError('Both registered critics must have passed the TRAIN screen')
    with np.load(args.diagnostic/'graded/predictions.npz', allow_pickle=False) as archive:
        predictions = {k: archive[k].copy() for k in archive.files}
    original_args = SimpleNamespace(**{k: Path(v) for k, v in io.read(parent_args.parent/'registration.json')['paths'].items()})
    bundle = diagnostic.inputs.load_inputs(original_args)
    original_args.neural = Path(original_args.neural)
    bundle = diagnostic.fitted.bind_registered_input_identity(bundle, io.read(original_args.neural/'registration.json'))
    arrays, _ = diagnostic.extractor.verify_complete(original_args.neural, bundle, io.sha(original_args.neural/'registration.json'))
    original_done = diagnostic.fitted.verify_complete(parent_args.parent, io.sha(parent_args.parent/'registration.json'))
    model = pickle.loads(args.base.read_bytes())
    anchor = pickle.loads((parent_args.parent/'smooth32-review1.pkl').read_bytes())
    if io.structural_digest(model.mentor) != io.structural_digest(anchor.mentor):
        raise ValueError('Frozen hybrid mentor differs from the original TRAIN anchor')
    align = diagnostic.protocol.protocol.strict_records(parent_args.parent/'calibration-alignment.jsonl')
    with np.load(parent_args.parent/'calibration-oof.npz', allow_pickle=False) as archive:
        oof = {(r['id'], r['path']): p for r, p in zip(align, archive['probabilities'])}
    evidence = {**spec['input_evidence'], **{str((args.diagnostic/n).resolve()): h for n, h in done['digests'].items()}}
    evidence.update({str((parent_args.parent/n).resolve()): h for n, h in original_done['artifacts'].items()})
    for p in (args.base, args.diagnostic/'complete.json', args.diagnostic/'graded/summary.json',
              args.diagnostic/'graded/predictions.npz', parent_args.parent/'complete.json'):
        evidence[str(p.resolve())] = io.sha(p)
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]), 'input_evidence': evidence,
        'paths': {n: str(getattr(args, n).resolve()) for n in ('diagnostic', 'base', 'output')},
        'variants': ['option', 'combined'], 'smoothing': 32., 'training_review_questions': 320,
        'new_critic_questions': 256, 'quantiles': [.2, .4, .6, .8], 'calibration': 'Same fixed C=1 as screened diagnostic',
        'emissions': 'Use whole-question-held review calibration and solver forecasts',
        'math': 'Original mentor dynamics, observations and settings retained exactly',
        'base_and_transfer_unchanged': True, 'solver_budget': 2, 'maximum_reviews': 1,
        'new_encoder_calls': 0, 'new_cloud_calls': 0, 'development_or_test_read': False,
        'base_provenance': 'Previously selected fixed base artifact; no DEV questions or outcomes read here',
        'model_selection': False, 'controller_accuracy_claim': False}
    return rows, saved, predictions, bundle, arrays, oof, model, parent_args, record


def fit(args, bundle):
    rows, saved, predictions, training, arrays, oof, anchor, source, record = bundle
    regular = anchor.mentor.regular; recipe = io.read(Path(regular.classifier.directory)/'recipe.json')
    encoder_price = recipe['parameters']/1e9*recipe['max_length']/3.09
    old, _ = core.fit_mentor(training['rows'], training['trees'], training['judges'], copy.deepcopy(regular), oof,
                            training['metadata'], arrays, encoder_price, 32., True)
    if io.structural_digest(old) != io.structural_digest(anchor.mentor): raise ValueError('Original mentor does not reproduce exactly')
    lookup = {(r['id'], r['kind'], r['path']): v for r, v in zip(training['metadata'], arrays['state'])}
    labels = predictions['labels']; models = {}
    for name in ('option', 'combined'):
        x = predictions['option_logits'][:, None] if name == 'option' else np.stack(
            [predictions['old_logits'], predictions['option_logits']], axis=1)
        expected_labels = np.array([int(bool(training['judges'][r['id'], '']['answer']) and
            training['judges'][r['id'], '']['answer'] == r['answer']) for r in rows])
        if not np.array_equal(labels, expected_labels): raise ValueError('Critic labels/order changed')
        expected = diagnostic.crossfit(x, labels, diagnostic.fold_ids(rows))
        np.testing.assert_array_equal(expected, predictions[name])
        calibration = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000, random_state=diagnostic.SEED)).fit(x, labels)
        eligible = [i for i, r in enumerate(rows) if training['judges'][r['id'], '']['answer'] and training['judges'][r['id'], '']['parse_valid']]
        edges = np.quantile(predictions[name][eligible], [.2, .4, .6, .8])
        science_p = {r['id']: p for r, p in zip(rows, predictions[name])}; signals = {}
        for (qid, path), judge in training['judges'].items():
            if qid in science_p:
                signals[qid] = runtime.signal(judge, training['trees'][qid][''], science_p[qid], edges)
            else:
                probability = core.perception.softmax(lookup[qid, 'original-review', ''], regular.classifier.temperature)
                signals[qid] = core.mentor.signal(judge, training['trees'][qid][''], probability)
        paired = core.paired_records(training['rows'], training['trees'], training['judges'], oof, encoder_price, signals, True)
        for item in paired:
            if item['id'] in saved:
                new_cost = saved[item['id']]['token_slots']*record_parameters(source)/1e9/3.09
                item['review_cost'] += new_cost-(encoder_price if name == 'option' else 0.)
        mentor = copy.deepcopy(anchor.mentor); mentor.dynamics = core.reviewed_dynamics(regular, paired, 32.)
        mentor.dynamics['math'] = copy.deepcopy(anchor.mentor.dynamics['math'])
        mentor.fit_receipt = {**mentor.fit_receipt, 'review_critic': name, 'review_edges': edges.tolist(),
            'critic_calibration_question_excluded': True, 'critic_source_sha256': io.sha(args.diagnostic/'complete.json'),
            'registration_sha256': args.registration_sha256, 'paired_records_sha256': diagnostic.inputs.fingerprint(paired)}
        observer = diagnostic.option_model.from_checkpoint(source.verifier/'full', 'cpu')
        receipt = copy.deepcopy(anchor.fit_receipt)
        receipt['observer_recipe_sha256'] = sorted(set(receipt['observer_recipe_sha256']) | {runtime.original.asset_id(observer)})
        receipt.update(policy_module='critic_review_hybrid', review_critic=name, mentor=mentor.fit_receipt,
            registration_sha256=args.registration_sha256, original_mentor_exactly_reproduced=True,
            current_math_exactly_preserved=True, critic_training_questions=256)
        model = runtime.Planner(copy.deepcopy(anchor.base), mentor, copy.deepcopy(anchor.transfer), anchor.base_module,
            copy.deepcopy(anchor.base_settings), receipt, critic=runtime.ReviewCritic(observer, calibration, edges, name == 'combined'))
        path = args.output/(name+'.pkl'); io.atomic_bytes(path, pickle.dumps(model, protocol=5))
        models[name] = {'model_sha256': io.sha(path), 'fit_receipt': receipt, 'edges': edges.tolist()}
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256, 'models': models,
        'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')},
        'new_encoder_calls': 0, 'new_cloud_calls': 0, 'development_or_test_read': False})
    print({'complete': True, 'models': list(models), 'original_mentor_exactly_reproduced': True}, flush=True)


def record_parameters(source): return io.read(source.verifier/'full/recipe.json')['parameters']


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'fit'))
    for n in ('diagnostic', 'base', 'output'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with diagnostic.guarded.training_only_reads(), diagnostic.guarded.guarded.no_model_calls(), threadpool_limits(limits=1):
        bundle = inputs(args); record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered fit')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json', '.run.lock') for p in args.output.iterdir()): raise ValueError('Preserve prior fit outputs')
            fit(args, bundle)


if __name__ == '__main__': main()
