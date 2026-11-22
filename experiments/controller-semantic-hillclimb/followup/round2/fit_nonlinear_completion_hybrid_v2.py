"""Fit only the TRAIN truth-preserving transfer for the richer Complete hybrid."""
import argparse
from pathlib import Path
import pickle
import sys

import nonlinear_completion_hybrid_v2 as runtime
import completion_mentor_transfer as transfer
import expanded_mentor_io as io
import completion_v4_diagnostic as guarded

SCHEMA = 'nonlinear-Complete16-mentor8-transfer-fit-v2'


def inputs(args):
    receipt = io.read(args.preparation/'preparation.json')
    path = args.preparation/'private-transfer-records.jsonl'
    if receipt['records_file_sha256'] != io.sha(path) or receipt['actual_transfer_fit_performed']:
        raise ValueError('Require the immutable, unfitted full256 TRAIN preparation')
    io.verify(receipt['source_files']); io.verify(receipt['input_evidence'])
    records = [io.json.loads(line) for line in path.read_text().splitlines()]
    if transfer.fingerprint(records) != receipt['records_sha256']: raise ValueError('Changed transfer labels')
    a = pickle.loads(args.a_model.read_bytes()); c = pickle.loads(args.c_model.read_bytes())
    if (a.fit_receipt.get('state_count') != 16 or a.feature_family != 'raw20_plus_project256'
            or not a.fit_receipt['provenance'].get('nonlinear_categorical_forecaster')):
        raise ValueError('Require the independently trained dense forest Complete16 model')
    evidence = {**receipt['input_evidence'], **{str(p.resolve()): io.sha(p) for p in
        (args.preparation/'preparation.json', path, args.a_model, args.c_model)}}
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]), 'input_evidence': evidence,
        'paths': {k: str(getattr(args, k).resolve()) for k in ('preparation', 'a_model', 'c_model', 'output')},
        'training_transfer_questions': 256, 'smoothing': 32., 'state_modes': [16, 8],
        'unchanged_parent_mentor_dynamics': True, 'unchanged_A_and_mentor_forecasters': True,
        'parent_C_previously_selected_on_development': True, 'new_fit_uses_TRAIN_labels_only': True,
        'new_neural_calls': 0, 'new_cloud_calls': 0, 'test_used': False,
        'maximum_solver_continuations': 2, 'maximum_logical_reviews': 1,
        'intended_scope': 'Incremental development improvement; current cached answer ceiling remains below final target'}
    return records, receipt, a, c, record


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'fit'))
    for name in ('preparation', 'a_model', 'c_model', 'output'): p.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with guarded.training_only_reads(), guarded.guarded.no_model_calls():
        records, receipt, a, c, record = inputs(args)
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered transfer fit')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json', '.run.lock') for p in args.output.iterdir()): raise ValueError('Preserve prior fits')
            kernels, fitted = transfer.fit(records, receipt)
            model = runtime.build(a, {'penalty': .02}, c, 'stronger_base_hybrid',
                {'by_subject': {'math': {'penalty': 0., 'threshold': 0.}}}, kernels, fitted,
                {'registration_sha256': args.registration_sha256})
            path = args.output/'hybrid.pkl'; io.atomic_bytes(path, pickle.dumps(model, protocol=5))
            io.verify(record['source_files']); io.verify(record['input_evidence'])
            io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
                'model_sha256': io.sha(path), 'fit_receipt': model.fit_receipt,
                'new_neural_calls': 0, 'new_cloud_calls': 0, 'test_used': False,
                'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})
            print({'complete': True, 'paired_training_questions': 256, 'state_modes': [16, 8]}, flush=True)


if __name__ == '__main__': main()
