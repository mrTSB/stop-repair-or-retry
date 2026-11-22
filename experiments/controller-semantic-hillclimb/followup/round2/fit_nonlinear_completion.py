"""Fit deployable joint categorical forests after the nested TRAIN screen."""
import argparse
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

import diagnose_nonlinear_completion as diagnostic
import fit_auxiliary_completion as previous

io, fixed, nested, runtime = previous.io, previous.fixed, previous.nested, previous.runtime
SCHEMA = 'expanded768-nonlinear-Complete16-fit-v1'


def inputs(args):
    spec = io.read(args.diagnostic/'registration.json'); done = io.read(args.diagnostic/'complete.json')
    source = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()})
    data, _, expected = diagnostic.inputs(source)
    if (spec != expected or done['schema'] != diagnostic.SCHEMA
            or done['registration_sha256'] != io.sha(args.diagnostic/'registration.json')):
        raise ValueError('Changed completed nonlinear diagnostic')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    io.verify({str(args.diagnostic/n): h for n, h in done['digests'].items()})
    summary = io.read(args.diagnostic/'summary.json')
    if not all(summary['reports'][f]['screen_passed'] for f in nested.FAMILIES): raise ValueError('Both families must pass')
    rows, trees, observer, aligned, _ = previous.inputs(SimpleNamespace(features=source.features, output=args.output))
    if data['records'] != aligned['records']: raise ValueError('Changed feature row alignment')
    with np.load(args.diagnostic/'private-oof.npz', allow_pickle=False) as archive:
        predictions = {k: archive[k].copy() for k in nested.FAMILIES}
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**spec['input_evidence'], **{str((args.diagnostic/n).resolve()): h for n, h in done['digests'].items()},
                           str((args.diagnostic/'complete.json').resolve()): io.sha(args.diagnostic/'complete.json')},
        'paths': {k: str(getattr(args, k).resolve()) for k in ('diagnostic', 'output')},
        'families': list(nested.FAMILIES), 'training_questions': 768, 'smoothing': 32.,
        'emissions': 'Reuse nested outer-held question predictions; final forecaster reselects with inner TRAIN folds',
        'new_neural_calls': 0, 'cloud_calls': 0, 'development_or_test_read': False, 'controller_accuracy_claim': False}
    return rows, trees, observer, data, predictions, record


def fit(args, rows, trees, observer, data, predictions, record):
    models = {}
    for family in nested.FAMILIES:
        chosen, scores = diagnostic.forecast.select(data, family, np.ones(len(data['labels']), dtype=bool))
        forecaster = diagnostic.forecast.fit(data['features'][family], data['labels'], chosen)
        obs = runtime.DenseObserver(observer.directory, observer.temperature, calibration=observer.calibration) if family == nested.FAMILIES[1] else observer
        obs._recipe = observer._recipe
        model = runtime.fit_dynamics(rows, trees, obs, data['records'], predictions[family], forecaster, family,
            {'registration_sha256': args.registration_sha256, 'nested_diagnostic_sha256': io.sha(args.diagnostic/'complete.json'),
             'final_settings': chosen, 'final_inner_search': scores, 'nonlinear_categorical_forecaster': True})
        path = args.output/(family+'.pkl'); io.atomic_bytes(path, pickle.dumps(model, protocol=5))
        models[family] = {'model_sha256': io.sha(path), 'fit_receipt': model.fit_receipt}
        print({'fitted': family, 'selected': chosen}, flush=True)
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'models': models, 'training_questions': 768, 'new_neural_calls': 0, 'cloud_calls': 0,
        'development_or_test_read': False, 'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'fit'))
    for n in ('diagnostic', 'output'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with fixed.training_only_reads(), fixed.guarded.no_model_calls(), threadpool_limits(limits=1):
        bundle = inputs(args); record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered fit')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json', '.run.lock') for p in args.output.iterdir()): raise ValueError('Preserve prior fits')
            fit(args, *bundle)


if __name__ == '__main__': main()
