"""Fit the screened semantic-correctness POMDP without new inference."""
import argparse
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

import diagnose_semantic_current as diagnostic
import semantic_completion_pomdp as runtime

io, fixed = diagnostic.io, diagnostic.fixed
SCHEMA = 'semantic-current-Complete16-fit-v1'


def inputs(args):
    spec = io.read(args.diagnostic/'registration.json'); done = io.read(args.diagnostic/'complete.json')
    source = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()})
    data, original, indices, records, logits, baseline, expected = diagnostic.inputs(source)
    if (expected != spec or done['registration_sha256'] != io.sha(args.diagnostic/'registration.json')
            or not done['screen_passed']): raise ValueError('Require the unchanged passing nested diagnostic')
    io.verify(spec['source_files']); io.verify(spec['input_evidence']); io.verify({str(args.diagnostic/n): h for n, h in done['digests'].items()})
    with np.load(args.diagnostic/'private-oof.npz', allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['indices'], indices); np.testing.assert_array_equal(archive['parent'], baseline)
        np.testing.assert_array_equal(archive['semantic_logits'], logits); corrected = archive['corrected'].copy()
    parent_args = SimpleNamespace(**{k: Path(v) for k, v in io.read(source.models/'registration.json')['paths'].items()})
    rows, trees, _, aligned, _, _ = diagnostic.forest_source.inputs(parent_args)
    if aligned['records'] != data['records']: raise ValueError('Changed paired data ordering')
    qids = {r['id'] for r in records}; rows = [r for r in rows if r['id'] in qids]
    if len(rows) != 256: raise ValueError('Exactly original256 science training rows required')
    base = pickle.loads((source.models/(diagnostic.FAMILY+'.pkl')).read_bytes())
    critic_spec = io.read(source.critic/'registration.json'); directory = Path(critic_spec['paths']['verifier'])/'full'
    semantic = diagnostic.critic.option_model.from_checkpoint(directory, 'cpu'); semantic._recipe = io.read(directory/'recipe.json')
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**spec['input_evidence'], **{str((args.diagnostic/n).resolve()): h for n, h in done['digests'].items()},
                           str((args.diagnostic/'complete.json').resolve()): io.sha(args.diagnostic/'complete.json')},
        'paths': {n: str(getattr(args, n).resolve()) for n in ('diagnostic', 'output')},
        'original_science_questions': 256, 'transition_questions': 768, 'same_parent_forecaster': True,
        'calibrator': 'Fixed standardized logistic C=0.1; all original valid TRAIN nodes, parent OOF forecasts',
        'emissions': 'Use fully nested held original-question sensor posteriors', 'new_neural_calls': 0,
        'cloud_calls': 0, 'development_or_test_used': False, 'state_count': 16, 'exact_finite_Bellman': True,
        'new_sensor_price': 'Conservative512-slot expectation in planning, actual trimmed slots in runtime ledger'}
    return rows, trees, data, indices, records, logits, baseline, corrected, base, semantic, record


def fit(args, rows, trees, data, indices, records, logits, baseline, corrected, base, semantic, record):
    valid = data['valid'][indices]; selected = [r for r, flag in zip(records, valid) if flag]
    x = diagnostic.correction.features(selected, baseline[valid], logits[valid]); y = data['labels'][indices][valid] >> 3
    calibrator = diagnostic.correction.fit(x, y)
    model = runtime.refit(base, rows, trees, records, corrected, semantic, calibrator,
                         {'registration_sha256': args.registration_sha256, 'diagnostic_sha256': io.sha(args.diagnostic/'complete.json')})
    path = args.output/'semantic.pkl'; io.atomic_bytes(path, pickle.dumps(model, protocol=5))
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'model_sha256': io.sha(path), 'fit_receipt': model.fit_receipt, 'new_neural_calls': 0, 'cloud_calls': 0,
        'development_or_test_used': False, 'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})
    print({'complete': True, 'training_sensor_questions': 256, 'transition_questions': 768}, flush=True)


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
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered sensor fit')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json', '.run.lock') for p in args.output.iterdir()): raise ValueError('Preserve prior fits')
            fit(args, *bundle)


if __name__ == '__main__': main()
