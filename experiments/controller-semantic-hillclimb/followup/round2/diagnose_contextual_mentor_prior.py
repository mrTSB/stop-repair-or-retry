"""Nested TRAIN screen of a context-dependent prior before buying an LLM review."""
import argparse
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

import contextual_mentor_prior as context
import fit_nonlinear_completion as parent

io, fixed, nested = parent.io, parent.fixed, parent.nested
SCHEMA = 'contextual-mentor-prior-TRAIN-diagnostic-v1'


def inputs(args):
    receipt = io.read(args.preparation/'preparation.json')
    path = args.preparation/'private-transfer-records.jsonl'
    if receipt['records_file_sha256'] != io.sha(path) or receipt['actual_transfer_fit_performed']:
        raise ValueError('Require original256 unfitted transfer preparation')
    io.verify(receipt['source_files']); io.verify(receipt['input_evidence'])
    records = [io.json.loads(line) for line in path.read_text().splitlines()]
    if context.transfer.fingerprint(records) != receipt['records_sha256']: raise ValueError('Changed transfer label identity')
    spec = io.read(args.base_models/'registration.json')
    source = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()})
    _, _, _, all_data, predictions, expected = parent.inputs(source)
    if spec != expected: raise ValueError('Changed nested A training fit')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    lookup = {r['id']: i for i, r in enumerate(all_data['records']) if r['path'] == ''}
    indices = np.asarray([lookup[r['id']] for r in records]); selected = [all_data['records'][i] for i in indices]
    if len(records) != 256 or any((all_data['labels'][i] >> 3) != context.CURRENT[r['target']] for i, r in zip(indices, records)):
        raise ValueError('Exactly256 aligned root labels required')
    a = predictions[nested.FAMILIES[1]][indices]
    data = {'records': selected, 'features': {k: v[indices] for k, v in all_data['features'].items()},
        'labels': np.asarray([r['target'] for r in records]), 'A_oof': a,
        'current_mass': a @ context.transfer.plain.CURRENT, 'transfer_records': records}
    evidence = {**receipt['input_evidence'], **spec['input_evidence'],
        str((args.preparation/'preparation.json').resolve()): io.sha(args.preparation/'preparation.json'),
        str(path.resolve()): io.sha(path), str((args.base_models/'registration.json').resolve()): io.sha(args.base_models/'registration.json')}
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]), 'input_evidence': evidence,
        'paths': {k: str(getattr(args, k).resolve()) for k in ('preparation', 'base_models', 'output')},
        'questions': 256, 'families': list(nested.FAMILIES), 'regularization': list(context.CS),
        'records_sha256': fixed.fingerprint(selected), 'label_sha256': fixed.array_sha(data['labels']),
        'A_forecasts_sha256': fixed.array_sha(a), 'outer_folds': 3, 'inner_folds': 3,
        'comparison': 'Original fixed transfer also refitted without each outer-held question group',
        'current_truth_mass': 'Exactly preserved from independently excluded A16 forecasts',
        'no_stacked_feature_leakage': 'Context classifier uses raw public features only; no label-derived A forecasts enter classifier fitting',
        'screen': 'Lower joint8 loss AND mentor-correctness Brier than old held-fold transfer',
        'new_neural_calls': 0, 'new_cloud_calls': 0, 'development_or_test_read': False,
        'controller_accuracy_claim': False, 'runtime_cost': 'Same paid root observer forward; CPU classifier time reported separately'}
    return data, record


def diagnose(args, data, record):
    reports, outputs, fold_records = {}, {}, {}; draws = np.random.default_rng(238031).integers(0, 256, (2000, 256))
    for family in nested.FAMILIES:
        p, old = np.full((256, 8), np.nan), np.full((256, 8), np.nan); fold_records[family] = []
        for fold in range(3):
            held, predicted, baseline, model, receipt = context.outer_prediction(data, family, fold)
            p[held] = predicted; old[held] = baseline; fold_records[family].append(receipt)
            io.atomic_bytes(args.output/(family+f'-outer{fold}.pkl'), pickle.dumps(model, protocol=5))
            print({'family': family, 'fold': fold, 'C': receipt['C']}, flush=True)
        if not np.isfinite(p).all() or not np.isfinite(old).all(): raise ValueError('Incomplete held-root forecasts')
        np.testing.assert_allclose(p @ context.CURRENT, data['current_mass'], atol=1e-12)
        labels = data['labels']; target = labels % 2
        before = {'joint_log_loss': context.loss(labels, old), 'mentor_Brier': (old @ context.MENTOR-target)**2}
        after = {'joint_log_loss': context.loss(labels, p), 'mentor_Brier': (p @ context.MENTOR-target)**2}
        compared = {}
        for metric in before:
            delta = after[metric]-before[metric]
            compared[metric] = {'old_transfer': float(before[metric].mean()), 'contextual': float(after[metric].mean()),
                'delta': float(delta.mean()), 'paired_question_CI95': np.quantile(delta[draws].mean(1), [.025, .975]).tolist()}
        reports[family] = {'comparisons': compared, 'screen_passed': all(v['delta'] < 0 for v in compared.values())}
        outputs[family] = p; outputs[family+'_old'] = old
    io.npz(args.output/'private-oof.npz', outputs); io.write(args.output/'fold-receipts.json', fold_records)
    io.write(args.output/'summary.json', {'schema': SCHEMA, 'reports': reports, 'questions': 256,
        'new_neural_calls': 0, 'new_cloud_calls': 0, 'development_or_test_read': False, 'controller_accuracy_claim': False})
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'any_screen_passed': any(r['screen_passed'] for r in reports.values()),
        'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})
    print({'reports': reports}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'diagnose'))
    for n in ('preparation', 'base_models', 'output'): p.add_argument('--'+n.replace('_', '-'), type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with fixed.training_only_reads(), fixed.guarded.no_model_calls(), threadpool_limits(limits=1):
        data, record = inputs(args)
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered contextual-prior study')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json', '.run.lock') for p in args.output.iterdir()): raise ValueError('Preserve earlier diagnostic')
            diagnose(args, data, record)


if __name__ == '__main__': main()
