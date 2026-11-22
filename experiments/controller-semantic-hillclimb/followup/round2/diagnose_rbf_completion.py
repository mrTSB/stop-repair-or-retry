"""Registered nested TRAIN test of rbf joint-state observation forecasts."""
import argparse
from pathlib import Path
import pickle
import sys

import numpy as np
from threadpoolctl import threadpool_limits

import diagnose_factorized_completion as source
import rbf_completion_forecast as forecast

io, fixed, nested = source.io, source.fixed, source.nested
SCHEMA = 'expanded768-rbf-joint-belief-TRAIN-v1'
FAMILIES = (nested.FAMILIES[1],)


def inputs(args):
    data, _, parent = source.inputs(args)
    spec = io.read(args.forest/'registration.json'); done = io.read(args.forest/'complete.json')
    if (spec['schema'] != 'expanded768-nonlinear-joint-belief-TRAIN-v1'
            or done['registration_sha256'] != io.sha(args.forest/'registration.json')
            or spec['paths']['features'] != str(args.features.resolve())
            or spec['input_evidence'] != parent['input_evidence']): raise ValueError('Changed retained forest diagnostic')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    io.verify({str(args.forest/n): h for n, h in done['digests'].items()})
    with np.load(args.forest/'private-oof.npz', allow_pickle=False) as archive:
        control = {k: archive[k].copy() for k in FAMILIES}
    evidence = {**parent['input_evidence'], **{str((args.forest/n).resolve()): h for n, h in done['digests'].items()},
        str((args.forest/'complete.json').resolve()): io.sha(args.forest/'complete.json')}
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': evidence, 'paths': {k: str(getattr(args, k).resolve()) for k in ('features', 'categorical', 'forest', 'output')},
        'training_questions': 768, 'decision_nodes': 2304, 'families': list(FAMILIES),
        'settings': list(forecast.SETTINGS), 'random_fourier_components': forecast.COMPONENTS, 'prior_weight': forecast.PRIOR_WEIGHT,
        'selection': 'Three inner whole-question folds minimize joint log loss, three outer folds',
        'screen': 'Lower joint log loss AND mean-four marginal Brier than retained dense ExtraTrees model',
        'bootstrap_draws': 2000, 'bootstrap_seed': 238031, 'new_neural_calls': 0,
        'cloud_calls': 0, 'development_or_test_read': False, 'controller_accuracy_claim': False}
    return data, control, record


def diagnose(args, data, controls, record):
    with io.lock(args.output/'.run.lock'), threadpool_limits(limits=1):
        if any(p.name not in ('registration.json', '.run.lock') for p in args.output.iterdir()):
            raise ValueError('Preserve partial/completed diagnostic')
        groups = [[i for i, r in enumerate(data['records']) if r['id'] == qid]
                  for qid in sorted({r['id'] for r in data['records']})]
        if len(groups) != 768 or any(len(g) != 3 for g in groups): raise ValueError('Complete question groups required')
        draws = np.random.default_rng(238031).integers(0, 768, (2000, 768))
        old = fixed.losses(data['labels'], controls[nested.FAMILIES[1]])
        oof, receipts, reports = {}, {}, {}
        for family in FAMILIES:
            p = np.full((2304, 16), np.nan); receipts[family] = []
            for fold in range(3):
                held, value, model, receipt = forecast.outer_prediction(data, family, fold)
                p[held] = value; receipts[family].append(receipt)
                io.atomic_bytes(args.output/(family+f'-outer{fold}.pkl'), pickle.dumps(model, protocol=5))
                print({'family': family, 'outer_fold_complete': fold, 'selected': receipt['selected']}, flush=True)
            if not np.isfinite(p).all(): raise ValueError('Incomplete question-excluded forecasts')
            oof[family] = p; new = fixed.losses(data['labels'], p); comparison = {}
            for metric in old:
                delta = np.array([(new[metric][g]-old[metric][g]).mean() for g in groups])
                comparison[metric] = {'retained_dense_forest': float(old[metric].mean()),
                    'rbf': float(new[metric].mean()), 'delta': float(delta.mean()),
                    'paired_question_CI95': np.quantile(delta[draws].mean(1), [.025, .975]).tolist()}
            reports[family] = {'comparisons': comparison, 'screen_passed': bool(
                comparison['joint_log_loss']['delta'] < 0 and comparison['mean_four_marginal_Brier']['delta'] < 0)}
        io.npz(args.output/'private-oof.npz', oof); io.write(args.output/'fold-receipts.json', receipts)
        io.write(args.output/'summary.json', {'schema': SCHEMA, 'reports': reports, 'questions': 768,
            'new_neural_calls': 0, 'cloud_calls': 0, 'development_or_test_read': False, 'controller_accuracy_claim': False,
            'paired_CI_scope': 'Descriptive, fitted folds fixed'})
        io.verify(record['source_files']); io.verify(record['input_evidence'])
        io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
            'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')},
            'any_screen_passed': any(v['screen_passed'] for v in reports.values())})
        print({'reports': reports}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'diagnose'))
    for n in ('features', 'categorical', 'forest', 'output'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with fixed.training_only_reads(), fixed.guarded.no_model_calls():
        data, controls, record = inputs(args)
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered study')
        diagnose(args, data, controls, record)


if __name__ == '__main__': main()
