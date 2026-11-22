"""Fit the screened math-only forecaster and matched finite observation kernel."""
import argparse
import copy
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

import math_specialization_diagnostic as diagnostic

io, core, guarded = diagnostic.io, diagnostic.core, diagnostic.guarded
SCHEMA = 'math-specialized-eight-state-fit-v1'


def inputs(args):
    spec = io.read(args.diagnostic/'registration.json'); done = io.read(args.diagnostic/'complete.json')
    old_args = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()})
    data, expected = diagnostic.inputs(old_args)
    if (spec != expected or done['schema'] != diagnostic.SCHEMA or not done['screen_passed']
            or done['registration_sha256'] != io.sha(args.diagnostic/'registration.json')):
        raise ValueError('Require unchanged successful nested math diagnostic')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    io.verify({str(args.diagnostic/n): h for n, h in done['digests'].items()})
    original = io.read(old_args.base_fit/'registration.json')
    path = Path(original['paths']['base'])/'C0.1-smooth32.pkl'
    base = pickle.loads(path.read_bytes())
    if (type(base) is not core.Planner or base.fit_receipt['paired_questions'] != 512
            or base.fit_receipt['calibration_C'] != .1 or base.fit_receipt['smoothing'] != 32.
            or base.fit_receipt['state_bits'] != ['current_correct', 'repair_correct', 'retry_correct']):
        raise ValueError('Require original pooled eight-state POMDP ancestry')
    with np.load(args.diagnostic/'private-oof.npz', allow_pickle=False) as archive:
        indices, oof = archive['indices'].copy(), archive['specialized'].copy()
    math = np.array([r['subject'] == 'math' for r in data['records']])
    if not np.array_equal(indices, np.flatnonzero(math)) or oof.shape != (768, 8):
        raise ValueError('Math OOF alignment changed')
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**spec['input_evidence'], str(path.resolve()): io.sha(path),
            **{str((args.diagnostic/n).resolve()): h for n, h in done['digests'].items()},
            str((args.diagnostic/'complete.json').resolve()): io.sha(args.diagnostic/'complete.json')},
        'paths': {n: str(getattr(args, n).resolve()) for n in ('diagnostic', 'output')},
        'math_questions': 256, 'decision_nodes': 768, 'final_selection': 'Same inner whole-question joint-loss search on all math TRAIN',
        'emission_forecasts': 'Separate nested outer-held math predictions, never final selection outputs',
        'emission_smoothing': .1, 'emission_cohort': 'Math TRAIN only',
        'structural_prior_transition_cost': 'Exact retained pooled-model math arrays, unchanged',
        'new_neural_calls': 0, 'cloud_calls': 0, 'development_or_test_read': False}
    return data, base, math, oof, record


def fit(data, base, math, oof, registration):
    setting, search = diagnostic.select(data, math)
    forecaster = diagnostic.fit(data['x'][math], data['y'][math], setting)
    support = np.ones((2, 8, core.N_OBS)); support[:, core.CURRENT == 1, -1] = 0.
    counts = .1*support
    for i, index in enumerate(np.flatnonzero(math)):
        row = data['records'][index]
        if not row['path']: continue
        signal = core.observation(oof[i], data['valid'][index])
        counts[('repair', 'retry').index(row['path']), data['y'][index], signal] += 1
    d = copy.deepcopy(base.dynamics['math']); d.emission = counts/counts.sum(2, keepdims=True)
    receipt = {**base.fit_receipt, 'schema': SCHEMA, 'observation_forecast_math_questions': 256,
        'structural_dynamics_pooled_training_questions': 512, 'forecaster_settings': setting,
        'final_inner_search': search, 'nested_emission_forecasts': True, 'emission_fit_math_only': True,
        'prior_transition_cost_unchanged': True, 'registration_sha256': registration}
    return core.Planner(copy.deepcopy(base.classifier), forecaster, {'math': d}, receipt)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'fit'))
    for n in ('diagnostic', 'output'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with guarded.training_only_reads(), guarded.guarded.no_model_calls(), threadpool_limits(limits=1):
        data, base, math, oof, record = inputs(args)
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered math fit')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json', '.run.lock') for p in args.output.iterdir()): raise ValueError('Preserve prior fit')
            model = fit(data, base, math, oof, args.registration_sha256)
            path = args.output/'math.pkl'; io.atomic_bytes(path, pickle.dumps(model, protocol=5))
            io.verify(record['source_files']); io.verify(record['input_evidence'])
            io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
                'model_sha256': io.sha(path), 'fit_receipt': model.fit_receipt, 'new_neural_calls': 0, 'cloud_calls': 0,
                'development_or_test_read': False, 'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})
            print({'complete': True, 'selected': model.fit_receipt['forecaster_settings']}, flush=True)


if __name__ == '__main__': main()
