"""Register a CPU-only science Complete-action POMDP fit on sealed TRAIN pairs.

One fixed forecast recipe is fitted. No development/test records or neural
inference are used. A later separate evaluation must retain the strongest
previous A and every unchanged fixed baseline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from threadpoolctl import threadpool_limits

import completion_pomdp as policy
import development_nonlinear as cached
import development_repairability as original
import expanded_mentor_io as io
import training_sparse_text as guarded
from prepare_mixed_mentor import plain_data

SCHEMA = 'science-terminal-completion-fit-v1'
REGULARIZATION = .1
SMOOTHING = 32.


def load_inputs(args):
    import science_complete_actions as actions
    rows, trees, completion_receipt = actions.training_data(args.data, args.complete)
    if len(rows) != 256 or {r['subject'] for r in rows} != {'chemistry', 'physics'}:
        raise ValueError('All256 registered TRAIN science questions are required')
    if any(sum(r['subject'] == s for r in rows) != 128 for s in ('chemistry', 'physics')):
        raise ValueError('Changed science training subject mix')
    ordinary, plain = plain_data(args.data, 'train')
    if len(ordinary) != 512 or {r['id'] for r in rows} != {r['id'] for r in ordinary if r['subject'] != 'math'}:
        raise ValueError('Complete grid omitted or replaced a registered science question')
    registered_rows = {r['id']: r for r in ordinary}
    if any(row != registered_rows[row['id']] for row in rows):
        raise ValueError('A science question or training reference changed')
    base_identity = io.read(args.base/'identity.json')
    train_sha = io.sha(args.data/'splits/train-questions.jsonl')
    if (base_identity['new_train_questions_sha256'] != train_sha
        or base_identity['encoder_trained_on_new_fit_questions']
        or base_identity['source_sha256'] != io.sha(original.__file__)
        or base_identity['pomdp_sha256'] != io.sha(policy.plain.__file__)):
        raise ValueError('Require exact original unseen-training neural provenance')
    observer = cached.observer_from(args.base)
    if io.sha(args.legacy_examples) != base_identity['old_only_encoder_inputs_sha256']:
        raise ValueError('Old observer training examples changed')
    if observer._recipe['data_sha256'] != io.sha(args.legacy_examples):
        raise ValueError('Observer recipe trained on different examples')
    legacy_ids = {json.loads(line)['id'] for line in args.legacy_examples.read_text().splitlines()}
    if legacy_ids & {r['id'] for r in ordinary}:
        raise ValueError('New training question was used in the observer')
    metadata = [json.loads(line) for line in (args.base/'training-neural.jsonl').read_text().splitlines()]
    with np.load(args.base/'training-neural.npz', allow_pickle=False) as outputs:
        lookup, alignment = cached.verify_features(ordinary, plain, observer, metadata, outputs)
    for row in rows:
        for path, node in plain[row['id']].items():
            if trees[row['id']][path] != node:
                raise ValueError('A plain control changed while adding Complete actions')
    old_available = sum(any(policy.plain.correct(node, row['answer']) for node in plain[row['id']].values())
                        for row in rows)
    new_available = sum(any(policy.plain.correct(node, row['answer']) for node in trees[row['id']].values())
                        for row in rows)
    if new_available - old_available < 16:
        raise ValueError('Prespecified TRAIN feasibility screen failed: fewer than16 newly solvable science questions')
    corpus = Path(guarded.perception.CORPUS)
    if io.sha(corpus/'passages.jsonl') != observer._recipe['corpus_sha256']:
        raise ValueError('Frozen observer retrieval corpus changed')
    evidence = [args.base/'identity.json', args.base/'C1-smooth8.pkl',
        args.base/'training-neural.jsonl', args.base/'training-neural.npz',
        args.data/'splits/train-questions.jsonl', args.data/'train-plain-complete.json', args.data/'train-observations.jsonl',
        Path(observer.directory)/'recipe.json', args.legacy_examples, corpus/'passages.jsonl', corpus/'tfidf.pkl']
    evidence_files = {str(path.resolve()):io.sha(path) for path in evidence}
    evidence_files.update(completion_receipt['public_input_evidence']['files'])
    evidence_files.update(completion_receipt['source_files'])
    evidence_files[str((args.complete/'complete.json').resolve())] = completion_receipt['complete_sha256']
    evidence_files.update({str((args.complete/name).resolve()):digest
                           for name,digest in completion_receipt['digests'].items()})
    io.verify(evidence_files)
    identity = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'paths': {name:str(getattr(args,name).resolve()) for name in ('data','complete','base','legacy_examples','output')},
        'input_evidence': evidence_files,
        'complete_grid_receipt': completion_receipt, 'training_science_questions':256,
        'decision_nodes':768, 'original_neural_alignment_sha256': hashlib.sha256(
            json.dumps(alignment,sort_keys=True).encode()).hexdigest(),
        'training_headroom':{'plain7_oracle':old_available,'with_complete_oracle':new_available,
                            'novel':new_available-old_available,'minimum_novel':16,'policy_accuracy_claim':False},
        'regularization':REGULARIZATION, 'smoothing':SMOOTHING, 'models':1,
        'state_count':16, 'question_grouped_folds':3, 'maximum_additional_calls':2,
        'complete_terminal':True, 'math_unchanged':True, 'paid_scoring_features_used':False,
        'new_action_protocol':True, 'new_controller_result':False,
        'neural_calls':0, 'cloud_calls':0, 'development_or_test_records_read':False,
        'old_observer_trained_on_new_questions':False,
        'note':'A separate registered development study must compare all five controls and retain strongest A'}
    return rows, trees, observer, lookup, identity


def verify_complete(output):
    registration = io.read(output/'registration.json'); complete = io.read(output/'complete.json')
    if (complete['schema'] != SCHEMA or registration['schema'] != SCHEMA
        or complete['registration_sha256'] != io.sha(output/'registration.json')
        or complete['model_sha256'] != io.sha(output/'completion-A.pkl')
        or complete['training_science_questions'] != 256 or complete['neural_calls']
        or complete['cloud_calls'] or complete['development_or_test_records_read']):
        raise ValueError('Changed completed science Complete fit')
    io.verify(registration['source_files']); io.verify(registration['input_evidence'])
    restored = pickle.loads((output/'completion-A.pkl').read_bytes())
    if io.structural_digest(restored) != complete['structural_sha256']:
        raise ValueError('Saved learned POMDP state changed')
    return complete


def run(args):
    with guarded.no_model_calls(), threadpool_limits(limits=1):
        rows, trees, observer, lookup, identity = load_inputs(args)
        if args.mode == 'register':
            args.output.mkdir(parents=True, exist_ok=False)
            io.write(args.output/'registration.json', identity)
            print({'registered':True,'registration_sha256':io.sha(args.output/'registration.json')},flush=True)
            return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
            or io.read(args.output/'registration.json') != identity):
            raise ValueError('Fit requires unchanged explicit registration, inputs, source and recipe')
        with io.lock(args.output/'.fit.lock'):
            if (args.output/'complete.json').exists(): verify_complete(args.output); return
            if (args.output/'completion-A.pkl').exists():
                raise ValueError('Preserve unsealed fit artifact for inspection')
            model = policy.fit(rows, trees, observer, lookup, SMOOTHING, REGULARIZATION)
            model.fit_receipt['registration_sha256'] = args.registration_sha256
            raw = pickle.dumps(model, protocol=5)
            structural = io.structural_digest(model)
            if io.structural_digest(pickle.loads(raw)) != structural:
                raise ValueError('Archive roundtrip changed learned state')
            io.atomic_bytes(args.output/'completion-A.pkl', raw)
            io.verify(identity['source_files']); io.verify(identity['input_evidence'])
            # Validate the exact completed action grid again after fitting.
            _, _, _, _, after = load_inputs(args)
            if after != identity: raise ValueError('Training inputs changed during CPU fitting')
            receipt = {'schema':SCHEMA,'registration_sha256':args.registration_sha256,
                'model_sha256':io.sha(args.output/'completion-A.pkl'),'structural_sha256':structural,
                'training_science_questions':256,'decision_nodes':768,
                'fit_receipt':model.fit_receipt,'neural_calls':0,'cloud_calls':0,
                'development_or_test_records_read':False,'controller_accuracy_claim':False}
            io.write(args.output/'complete.json',receipt); verify_complete(args.output)
            print({'complete':True,'training_science_questions':256,'state_count':16,
                   'development_or_test_records_read':False},flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','fit'))
    for name in ('data','complete','base','legacy_examples','output'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--registration-sha256')
    run(parser.parse_args())
