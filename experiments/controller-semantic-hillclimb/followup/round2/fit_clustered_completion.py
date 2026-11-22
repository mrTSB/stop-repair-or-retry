"""TRAIN-only clustered observation refit for the sealed768-question forests."""
import argparse
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

import fit_nonlinear_completion as previous
import clustered_completion_pomdp as policy

io, fixed, nested = previous.io, previous.fixed, previous.nested
SCHEMA = 'expanded768-clustered-observation-Complete16-fit-v1'


def inputs(args):
    spec = io.read(args.models/'registration.json'); done = io.read(args.models/'complete.json')
    source = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()})
    rows, trees, _, data, forecasts, expected = previous.inputs(source)
    if (spec != expected or done['schema'] != previous.SCHEMA
            or done['registration_sha256'] != io.sha(args.models/'registration.json')):
        raise ValueError('Changed original forest fit')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    io.verify({str(args.models/n): h for n, h in done['digests'].items()})
    models = {}
    for family in nested.FAMILIES:
        path = args.models/(family+'.pkl')
        if io.sha(path) != done['models'][family]['model_sha256']: raise ValueError('Base model changed')
        models[family] = pickle.loads(path.read_bytes())
        if models[family].fit_receipt != done['models'][family]['fit_receipt']: raise ValueError('Base fit receipt changed')
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**spec['input_evidence'], **{str((args.models/n).resolve()): h for n, h in done['digests'].items()},
            str((args.models/'complete.json').resolve()): io.sha(args.models/'complete.json')},
        'paths': {k: str(getattr(args, k).resolve()) for k in ('models', 'output')},
        'training_questions': 768, 'families': list(nested.FAMILIES),
        'partition': 'KMeans over sqrt joint excluded-question TRAIN child forecasts',
        'valid_clusters': 24, 'invalid_clusters': 6, 'n_init': 10, 'max_iter': 100, 'seed': 2381124,
        'reference_labels_used_to_choose_partition': False, 'partition_hyperparameter_search': False,
        'unchanged': ['forecaster', 'prior', 'transition', 'cost', '16 states', '30 observations', 'exact finite Bellman backup'],
        'new_neural_calls': 0, 'cloud_calls': 0, 'development_or_test_read': False, 'controller_accuracy_claim': False}
    return rows, trees, data, models, forecasts, record


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'fit'))
    for n in ('models', 'output'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with fixed.training_only_reads(), fixed.guarded.no_model_calls(), threadpool_limits(limits=1):
        rows, trees, data, models, forecasts, record = inputs(args)
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed clustered observation registration')
        with io.lock(args.output/'.fit.lock'):
            if any(p.name not in ('registration.json', '.fit.lock') for p in args.output.iterdir()): raise ValueError('Preserve prior fits')
            results = {}
            for family, base in models.items():
                model = policy.rebin(base, rows, trees, data['records'], forecasts[family])
                for subject in base.dynamics:
                    for field in ('prior', 'transition', 'cost'):
                        np.testing.assert_array_equal(getattr(base.dynamics[subject], field), getattr(model.dynamics[subject], field))
                model.fit_receipt['kernel_registration_sha256'] = args.registration_sha256
                path = args.output/(family+'.pkl'); io.atomic_bytes(path, pickle.dumps(model, protocol=5))
                results[family] = {'model_sha256': io.sha(path), 'fit_receipt': model.fit_receipt}
            io.verify(record['source_files']); io.verify(record['input_evidence'])
            io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
                'models': results, 'training_questions': 768, 'new_neural_calls': 0, 'cloud_calls': 0,
                'development_or_test_read': False,
                'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})
            print({'complete': True, 'models': list(results)}, flush=True)


if __name__ == '__main__': main()
