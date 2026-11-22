"""Nested TRAIN screen of an additional paid science correctness observation.

The auxiliary science encoder excludes all256 original science questions. No
auxiliary training question is used to evaluate its in-sample knowledge. The
paired forest may use auxiliary trajectories, but its calibration predictors
are regenerated inside every held-question split. No neural or cloud call.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

import fit_nonlinear_completion as forest_source
import science_review_critic as critic
import semantic_current_belief as correction

io, fixed, nested = forest_source.io, forest_source.fixed, forest_source.nested
forecast = forest_source.diagnostic.forecast
SCHEMA = 'semantic-current-correction-nested-TRAIN-v1'
FAMILY = nested.FAMILIES[1]


def calibration_fold(qid):
    return int(hashlib.sha256(('semantic-current-inner-v1:'+qid).encode()).hexdigest()[:8], 16) % 3


def fitting_provenance(payload):
    """Decode only the known, top-level indented provenance object."""
    text = payload.decode('utf-8'); matches = list(re.finditer(r'(?m)^  "fit_receipt": (\{)', text))
    if len(matches) != 1: raise ValueError('Expected one exact top-level fitting provenance object')
    value, _ = json.JSONDecoder().raw_decode(text, matches[0].start(1))
    if not isinstance(value, dict) or value.get('test_used') is not False:
        raise ValueError('Invalid historical fitting provenance')
    return value


def inputs(args):
    spec = io.read(args.models/'registration.json'); done = io.read(args.models/'complete.json')
    source = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()})
    rows, trees, observer, data, baseline, expected = forest_source.inputs(source)
    if expected != spec or done['registration_sha256'] != io.sha(args.models/'registration.json'): raise ValueError('Frozen forest fit changed')
    io.verify(spec['source_files']); io.verify(spec['input_evidence']); io.verify({str(args.models/n): h for n, h in done['digests'].items()})
    critic_spec = io.read(args.critic/'registration.json')
    critic_args = SimpleNamespace(**{k: Path(v) for k, v in critic_spec['paths'].items()})
    original_rows, _, _, _, expected = critic.load_inputs(critic_args)
    if expected != critic_spec: raise ValueError('Final V3 original-question exclusion changed')
    io.verify(critic_spec['source_files']); io.verify(critic_spec['input_evidence'])
    identity = io.read(args.semantic/'identity.json')
    train_path = Path(io.read(critic_args.parent/'registration.json')['paths']['new_data'])/'splits/train-questions.jsonl'
    if (identity['train_question_sha256'] != io.sha(train_path) or identity['science_model'] != str(critic_args.verifier.resolve())
            or identity['test_used']): raise ValueError('V3 feature archive used different TRAIN data')
    for name, value in identity['source_sha256'].items():
        if io.sha(Path(__file__).with_name(name)) != value: raise ValueError('V3 feature source changed')
    # Project only fitting provenance; the surrounding historical DEV summary
    # contains selection metrics which this TRAIN diagnostic must not decode.
    provenance = fitting_provenance((args.semantic/'summary.json').read_bytes())
    if (provenance['science_neural_sha256'] != io.sha(args.semantic/'training-neural.npz')
            or provenance['auxiliary_recipe_sha256'] != io.sha(critic_args.verifier/'full/recipe.json')):
        raise ValueError('V3 training feature/weight binding changed')
    recipe = io.read(critic_args.verifier/'full/recipe.json'); metadata = [json.loads(v) for v in (args.semantic/'training-neural.jsonl').read_text().splitlines()]
    with np.load(args.semantic/'training-neural.npz', allow_pickle=False) as archive: state = archive['state'].copy()
    original_ids = {r['id'] for r in original_rows}; original = np.array([r['id'] in original_ids for r in data['records']])
    if original.sum() != 768 or len(original_ids) != 256 or state.shape != (768, 4): raise ValueError('Exactly the original256 science questions required')
    if not np.isfinite(state).all() or not np.allclose(state[:, [0, 2]], 0) or not np.array_equal(state[:, 1], state[:, 3]):
        raise ValueError('Expected binary-only V3 observation logits')
    by_id = {r['id']: r for r in original_rows}; lookup = {}
    for item, logits in zip(metadata, state):
        qid, path = item['id'], item['path']; row = {**by_id[qid], 'answer': ''}; node = trees[qid][path]
        entry = critic.option_model.option_entry(row, node['answer'], recipe['numeric_width'])
        if (item['prompt_sha256'] != node['prompt_sha256'] or item['input_sha256'] != critic.inputs.fingerprint(entry)
                or path not in ('', 'repair', 'retry') or (qid, path) in lookup): raise ValueError('V3 paid input alignment changed')
        lookup[qid, path] = float(critic.margin(logits)/recipe['temperature'])
    indices = np.flatnonzero(original); records = [data['records'][i] for i in indices]
    logits = np.array([lookup[r['id'], r['path']] for r in records])
    evidence = {**spec['input_evidence'], **critic_spec['input_evidence'],
        **{str((args.models/n).resolve()): h for n, h in done['digests'].items()},
        str((args.models/'complete.json').resolve()): io.sha(args.models/'complete.json')}
    for name in ('identity.json', 'summary.json', 'training-neural.jsonl', 'training-neural.npz'):
        evidence[str((args.semantic/name).resolve())] = io.sha(args.semantic/name)
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]), 'input_evidence': evidence,
        'paths': {n: str(getattr(args, n).resolve()) for n in ('models', 'critic', 'semantic', 'output')},
        'original_science_questions': 256, 'parent_paired_training_questions': 768, 'new_neural_calls': 0, 'cloud_calls': 0,
        'development_or_test_outcomes_decoded': False, 'development_summary_access': 'Only fitting-provenance fields projected; metrics skipped',
        'outer_folds': 3, 'calibration_inner_folds': 3, 'forecaster_inner_selection_folds': 3,
        'calibration_C': .1, 'calibration_selection': False, 'only_valid_answers_need_semantic_observer': True,
        'future_state_conditionals_preserved': True, 'screen': 'Lower joint log loss and current Brier, no worse mean-four Brier',
        'bootstrap_draws': 2000, 'bootstrap_seed': 238031, 'controller_accuracy_claim': False,
        'component_neural_exclusion': 'All original256 are excluded from both encoders; auxiliary512 never scored with its in-sample V3',
        'feature_sha256': fixed.array_sha(logits), 'source_forecast_diagnostic': str(source.diagnostic.resolve())}
    return data, original, indices, records, logits, baseline[FAMILY][indices], record


def outer_prediction(data, original, indices, records, logits, baseline, fold):
    outer = np.array([r['fold'] for r in data['records']]); inner = np.array([calibration_fold(r['id']) for r in data['records']])
    x, y, valid = data['features'][FAMILY], data['labels'], data['valid']
    held = outer == fold; training = ~held; calibrated_features = np.full((len(y), 16), np.nan); searches = []
    for inner_fold in range(3):
        fit_rows = training & (inner != inner_fold); calibration_rows = training & original & (inner == inner_fold)
        exclusion = nested.group_receipt(data['records'], fit_rows, held | calibration_rows)
        settings, search = forecast.select(data, FAMILY, fit_rows)
        model = forecast.fit(x[fit_rows], y[fit_rows], settings)
        calibrated_features[calibration_rows] = nested.predict(model, x[calibration_rows], valid[calibration_rows])
        searches.append({'inner_fold': inner_fold, 'selected': settings, 'search': search, 'question_exclusion': exclusion})
    train_local = training[indices]; held_local = held[indices]; good = train_local & valid[indices]
    if not np.isfinite(calibrated_features[indices][train_local]).all(): raise ValueError('Incomplete nested calibration forecasts')
    training_features = correction.features([records[i] for i in np.flatnonzero(good)], calibrated_features[indices][good], logits[good])
    calibrator = correction.fit(training_features, (y[indices][good] >> 3))
    old = baseline[held_local]; new_current = old @ correction.CURRENT; observed = valid[indices][held_local]
    if observed.any():
        held_records = [records[i] for i in np.flatnonzero(held_local)]
        values = correction.features([r for r, v in zip(held_records, observed) if v], old[observed], logits[held_local][observed])
        new_current[observed] = calibrator.predict_proba(values)[:, 1]
    return held_local, correction.update(old, new_current, observed), calibrator, {
        'outer_fold': fold, 'parent_and_calibration_exclusion_nested': True, 'inner_fits': searches}


def diagnose(args, data, original, indices, records, logits, baseline, record):
    y = data['labels']; predictions = np.full((len(indices), 16), np.nan); receipts = []
    for fold in range(3):
        held_local, values, calibrator, receipt = outer_prediction(data, original, indices, records, logits, baseline, fold)
        predictions[held_local] = values
        io.atomic_bytes(args.output/f'calibrator-outer{fold}.pkl', pickle.dumps(calibrator, protocol=5))
        receipts.append(receipt)
        print({'completed_outer_calibration_fold': fold}, flush=True)
    if not np.isfinite(predictions).all(): raise ValueError('Incomplete held forecasts')
    old = fixed.losses(y[indices], baseline); new = fixed.losses(y[indices], predictions)
    groups = [[i for i, r in enumerate(records) if r['id'] == qid] for qid in sorted({r['id'] for r in records})]
    draws = np.random.default_rng(238031).integers(0, len(groups), (2000, len(groups))); report = {}
    for key in old:
        delta = np.array([(new[key][g]-old[key][g]).mean() for g in groups]); report[key] = {
            'parent': float(old[key].mean()), 'corrected': float(new[key].mean()), 'delta': float(delta.mean()),
            'paired_question_CI95': np.quantile(delta[draws].mean(1), [.025, .975]).tolist()}
    passed = report['joint_log_loss']['delta'] < 0 and report['current_Brier']['delta'] < 0 and report['mean_four_marginal_Brier']['delta'] <= 0
    io.npz(args.output/'private-oof.npz', {'corrected': predictions, 'parent': baseline, 'semantic_logits': logits, 'indices': indices})
    io.write(args.output/'fold-receipts.json', receipts)
    io.write(args.output/'summary.json', {'schema': SCHEMA, 'reports': report, 'screen_passed': passed, 'questions': 256,
        'new_neural_calls': 0, 'cloud_calls': 0, 'development_or_test_outcomes_decoded': False, 'controller_accuracy_claim': False,
        'paired_CI_scope': 'Descriptive, fitted folds fixed', 'new_observation_cost': 'Must pay an additional V3 pass at every valid observed science answer'})
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}, 'screen_passed': passed})
    print({'screen_passed': passed, 'reports': report}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'diagnose'))
    for n in ('models', 'critic', 'semantic', 'output'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with fixed.training_only_reads(), fixed.guarded.no_model_calls(), threadpool_limits(limits=1):
        bundle = inputs(args); record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Registered semantic diagnostic changed')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json', '.run.lock') for p in args.output.iterdir()): raise ValueError('Preserve prior diagnostic')
            diagnose(args, *bundle)


if __name__ == '__main__': main()
