"""Two fixed TRAIN-only observation diagnostics for the 16-state Complete model.

Replaces only the twenty cached neural head logits. No neural model is loaded,
no policy is selected, and no DEV/TEST question file is opened. V4 neural OOF
and downstream question-grouped CV are explicitly NON-NESTED diagnostics.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

import completion_pomdp as core
import development_refined_observer as refined
import expanded_mentor_io as io
import fit_completion_pomdp as fitted
import training_sparse_text as guarded

SCHEMA = 'completion-observation-old-vs-V4-TRAIN-diagnostic-v1'
VARIANTS = ('old_V1_raw20', 'V4_question_excluded_raw20')
REGULARIZATION = .1
BOOTSTRAPS = 2000
SEED = 238031
EXCLUSION_NOTE = refined.EXCLUSION_NOTE


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    return fingerprint({'shape': list(a.shape), 'dtype': a.dtype.str,
        'data_sha256': hashlib.sha256(a.tobytes()).hexdigest()})


@contextmanager
def training_only_reads():
    """Fail closed if any nested helper tries to open protected split files."""
    from unittest.mock import patch
    import builtins
    import io as streams
    original_builtin, original_stream = builtins.open, streams.open
    def checked(function):
        def call(file, *args, **kwargs):
            if isinstance(file, (str, bytes, Path)):
                name = Path(file).name
                if name in ('dev-questions.jsonl', 'test-questions.jsonl'):
                    raise AssertionError('Diagnostic forbids DEV/TEST question reads')
            return function(file, *args, **kwargs)
        return call
    with patch.object(builtins, 'open', checked(original_builtin)), patch.object(streams, 'open', checked(original_stream)):
        yield


def v4_lookup(data, directory, rows):
    """Rebuild TRAIN examples and exact paid inputs; do not use load_verified()."""
    registration = io.read(directory/'training-registration.json')
    contract = io.read(directory/'contract.json')
    inputs = io.read(directory/'training-inputs.json')
    summary = io.read(directory/'training-summary.json')
    metadata_path = directory/'train.jsonl'
    if (contract != {**registration, 'data_sha256': sha(metadata_path)}
            or inputs['input_sha256'] != sha(metadata_path)
            or contract['trainer_sha256'] != sha(Path(refined.__file__).with_name('refine_stream.py'))
            or contract['question_split_sha256'] != sha(data/'splits/train-questions.jsonl')
            or contract['selection_sha256'] != sha(data/'splits/selection.json')
            or contract['new_questions'] != 512 or contract['folds'] != 3
            or summary.get('development_data_used') or summary.get('fresh_data_used')):
        raise ValueError('Changed V4 TRAIN contract or excluded-question training scope')
    ordinary = io.read(data/'train-ordinary-complete.json')
    evidence = {str(path.resolve()): sha(path) for path in (
        metadata_path, directory/'training-registration.json', directory/'contract.json',
        directory/'training-inputs.json', directory/'training-summary.json', directory/'oof.npz',
        data/'train-ordinary-complete.json', data/'splits/selection.json')}
    for part, filename in (('base', 'train-observations.jsonl'), ('late', 'train-late-observations.jsonl')):
        path = data/filename
        if sha(path) != ordinary[part+'_bank_sha256'] or sha(path) != inputs[part+'_bank_sha256']:
            raise ValueError('Changed V4 paired TRAIN response banks')
        evidence[str(path.resolve())] = sha(path)
    for fold in (0, 1, 2, 'full'):
        folder = directory/str(fold)
        recipe_path = folder/'recipe.json'; recipe = io.read(recipe_path)
        if recipe['fold'] != fold or any(recipe.get(k) != v for k, v in contract.items()):
            raise ValueError('Changed V4 fold checkpoint recipe')
        evidence[str(recipe_path.resolve())] = sha(recipe_path)
        for filename, digest in recipe['weights'].items():
            path = folder/filename
            if sha(path) != digest:
                raise ValueError('V4 archived fold weight changed')
            evidence[str(path.resolve())] = digest
    metadata = refined.read(metadata_path)
    if (summary['nodes'] != len(metadata) or summary['questions'] != len({r['id'] for r in metadata})
            or summary['question_grouped_folds'] != 3
            or any(r['fold'] != refined.fold_for(r['id']) for r in metadata)):
        raise ValueError('V4 complete OOF row/fold identity changed')
    expected = refined.examples(data, 'train')
    retriever = refined.Retriever()
    if retriever.sha256 != contract['corpus_sha256']:
        raise ValueError('V4 public retrieval corpus changed')
    by_id = {r['id']: r for r in rows}
    if len(by_id) != 512 or len(expected) != 17*512 or set(Counter(r['id'] for r in expected).values()) != {17}:
        raise ValueError('Require exact512-question/17-node V4 TRAIN inputs')
    contexts = {r['id']: retriever.context({**r, 'answer': ''})[0] for r in rows}
    for item in expected:
        item['left'] = refined.left_text({**by_id[item['id']], 'answer': ''}, contexts[item['id']])
    prompts = {}
    for suffix, kind in (('observations', 'original-review'), ('late-observations', 'late-review')):
        for item in refined.committed(data/('train-'+suffix+'.jsonl')):
            key = item['id'], ('solver' if item['kind']=='solver' else kind), item['path']
            if key in prompts:
                raise ValueError('Duplicate paired TRAIN prompt receipt')
            prompts[key] = item['response']['prompt_sha256']
    with np.load(directory/'oof.npz', allow_pickle=False) as arrays:
        lookup, alignment = refined.aligned_lookup(metadata, arrays, expected, prompts, set(by_id))
    return lookup, alignment, {'input_evidence': evidence, 'training_questions': 512,
        'verified_new_neural_nodes': len(alignment), 'decision_nodes': len(lookup),
        'alignment_sha256': fingerprint(alignment), 'nested_crossfit': False,
        'neural_fold_provenance': 'Each row comes from checkpoint excluding its question hash fold',
        'crossfit_limitation': EXCLUSION_NOTE}


def table(rows, trees, old, newer):
    records, features, labels, valid = [], {name: [] for name in VARIANTS}, [], []
    if (len(rows) != 256 or len({r['id'] for r in rows}) != 256
            or Counter(r['subject'] for r in rows) != {'chemistry':128, 'physics':128}):
        raise ValueError('Use every256 TRAIN science question')
    for row in rows:
        safe = {**row, 'answer': ''}
        for path in ('', 'repair', 'retry'):
            paid = {p: trees[row['id']][p] for p in ('', path)}
            prefix = path+'/' if path else ''
            target_paths = [path, *[prefix+a for a in ('repair', 'retry', 'complete')]]
            labels.append(core.state(*[core.plain.correct(trees[row['id']][p], row['answer']) for p in target_paths]))
            valid.append(bool(paid[path]['answer']))
            for name, lookup in zip(VARIANTS, (old, newer)):
                value = lookup[row['id'], path]
                if [np.asarray(v).shape for v in value[:3]] != [(4,), (8,), (8,)]:
                    raise ValueError('Require exactly4+8+8 paid neural logits')
                features[name].append(core.plain.observables(safe, path, paid, value))
            records.append({'id': row['id'], 'subject': row['subject'], 'path': path,
                'fold': refined.fold_for(row['id']), 'prompt_sha256': paid[path]['prompt_sha256'],
                'paid_response_sha256': fingerprint(paid[path])})
    features = {name: np.asarray(values, dtype=float) for name, values in features.items()}
    if (any(not np.isfinite(x).all() for x in features.values())
            or not np.array_equal(features[VARIANTS[0]][:,20:], features[VARIANTS[1]][:,20:])):
        raise ValueError('Only the twenty neural logits may differ')
    return {'records': records, 'features': features, 'labels': np.asarray(labels, dtype=int),
        'valid': np.asarray(valid, dtype=bool)}


def load_inputs(args):
    model_receipt = fitted.verify_complete(args.base_fit)
    original = io.read(args.base_fit/'registration.json')
    original_args = SimpleNamespace(**{name: Path(value) for name, value in original['paths'].items()})
    # The completed fitter already SHA-binds its entire ancestry. Re-entering
    # its acquisition gate would reopen DEV only to recompute a historical
    # panel hash. Rebuild TRAIN data directly from the exact sealed bytes here.
    import science_complete_actions as actions
    all_rows, all_trees = fitted.plain_data(original_args.data, 'train')
    rows = [r for r in all_rows if r['subject'] != 'math']
    public = [actions.pilot.protocol.public_row(r) for r in rows]
    trees = {r['id']: all_trees[r['id']].copy() for r in rows}
    grid = original_args.complete
    registered = io.read(grid/'registration.json'); sealed = io.read(grid/'complete.json')
    if (registered['selection'] != [{'id':r['id'],'subject':r['subject']} for r in rows]
            or str(original_args.data.resolve()) != registered['paths']['data']
            or str(grid.resolve()) != registered['paths']['output']
            or original['complete_grid_receipt']['complete_sha256'] != sha(grid/'complete.json')):
        raise ValueError('Completed fitter and TRAIN action grid differ')
    io.verify({str(grid/name): digest for name,digest in sealed['digests'].items()})
    pilot = Path(registered['paths']['pilot'])
    reused = actions.pilot.protocol.strict_records(pilot/'raw-generations.jsonl')
    if len(reused) != 192 or sha(pilot/'raw-generations.jsonl') != registered['raw_reuse']['sha256']:
        raise ValueError('Changed exact192 reused TRAIN observations')
    known, coverage = actions.restore(public, trees, {v['request_sha256'] for v in reused},
        actions.pilot.protocol.strict_records(grid/'train.jsonl'),
        actions.pilot.protocol.strict_records(grid/'raw-generations.jsonl'), True)
    if coverage != sealed['coverage']:
        raise ValueError('Changed full768 sealed TRAIN action grid')
    for row in rows:
        for prefix in actions.PREFIXES:
            trees[row['id']][actions.action_path(prefix)] = known[row['id'],prefix]['response']
    observer = fitted.cached.observer_from(original_args.base)
    metadata = [json.loads(line) for line in (original_args.base/'training-neural.jsonl').read_text().splitlines()]
    with np.load(original_args.base/'training-neural.npz', allow_pickle=False) as outputs:
        old, old_alignment = fitted.cached.verify_features(all_rows, all_trees, observer, metadata, outputs)
    if fingerprint(old_alignment) != original['original_neural_alignment_sha256']:
        raise ValueError('Old cached paid-input alignment differs from the completed fit')
    newer, alignment, new_receipt = v4_lookup(original_args.data, args.verifier, all_rows)
    data = table(rows, trees, old, newer)
    evidence = {**original['input_evidence'], **new_receipt['input_evidence'],
        str((args.base_fit/'registration.json').resolve()): sha(args.base_fit/'registration.json'),
        str((args.base_fit/'complete.json').resolve()): sha(args.base_fit/'complete.json'),
        str((args.base_fit/'completion-A.pkl').resolve()): model_receipt['model_sha256']}
    receipt = {'input_evidence': evidence, 'source_files': io.sources(sys.modules[__name__]),
        'paths': {n: str(getattr(args,n).resolve()) for n in ('base_fit','verifier','output')},
        'data_records_sha256': fingerprint(data['records']), 'labels_sha256': array_sha(data['labels']),
        'valid_sha256': array_sha(data['valid']),
        'feature_sha256': {name: array_sha(value) for name,value in data['features'].items()},
        'old_neural_alignment_sha256': original['original_neural_alignment_sha256'],
        'V4': new_receipt, 'V4_alignment_sha256': fingerprint(alignment)}
    return data, receipt


def specification(receipt):
    return {**receipt, 'schema': SCHEMA, 'variants': list(VARIANTS), 'training_questions':256,
        'decision_nodes':768, 'state_bits':['current_correct','repair_correct','retry_correct','complete_correct'], 'joint_classes':16,
        'model': {'C':REGULARIZATION, 'max_iter':1000, 'temperature':1., 'penalty':'l2', 'multiclass':'multinomial'},
        'feature_rule':'Same clipped raw20 then same paid visible/numeric features; only raw20 replaced',
        'row_weight':1., 'equal_question_mass':3.,
        'row_weight_reason':'Exactly three nodes per question; preserves original C.1 training normalization',
        'folds':'int(SHA256(question_id)[:8],16)%3; all three question nodes held together',
        'scaler_and_classifier_question_excluded':True, 'nested_crossfit':False, 'crossfit_limitation':EXCLUSION_NOTE,
        'metrics':['joint_log_loss','joint_Brier','current_Brier','repair_Brier','retry_Brier','Complete_Brier','mean_four_marginal_Brier'],
        'bootstrap':{'draws':BOOTSTRAPS,'seed':SEED,'unit':'question; all three nodes together','scope':'Descriptive with fitted folds fixed'},
        'screen':'V4 joint_log_loss lower AND mean_four_marginal_Brier lower; CI descriptive',
        'screen_scope':'Only motivates separately controlled runtime study; no A promotion or fresh claim',
        'temperature_search':False,'hyperparameter_search':False,'policy_or_threshold_search':False,
        'no_runtime_model_fit':True,'neural_calls':0,'cloud_calls':0,'development_or_test_read':False,
        'environment':{'python':list(sys.version_info[:3]),'packages':{n:importlib.metadata.version(n) for n in ('numpy','scikit-learn')}}}


def outer_prediction(data, name, fold):
    records=data['records']; held=np.asarray([r['fold']==fold for r in records]); train=~held
    train_ids={r['id'] for r,flag in zip(records,train) if flag}; held_ids={r['id'] for r,flag in zip(records,held) if flag}
    if not train_ids or not held_ids or train_ids & held_ids:
        raise ValueError('Whole-question exclusion failed')
    x, y=data['features'][name],data['labels']
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model=core.classifier_for(x[train],y[train],REGULARIZATION)
    fit_seconds=time.perf_counter()-started; started=time.perf_counter()
    probability=np.full((int(held.sum()),16),1e-9)
    probability[:,model.classes_.astype(int)]+=model.predict_proba(x[held])
    probability=np.asarray([core.mask_invalid(p,v) for p,v in zip(probability,data['valid'][held])])
    receipt={'fold':fold,'training_questions':len(train_ids),'held_questions':len(held_ids),
        'training_ids_sha256':fingerprint(sorted(train_ids)),'held_ids_sha256':fingerprint(sorted(held_ids)),
        'fit_seconds':fit_seconds,'prediction_seconds':time.perf_counter()-started,
        'warnings':[str(w.message) for w in captured], 'classes':model.classes_.astype(int).tolist()}
    return held,probability,model,receipt


def losses(y,p):
    target=np.eye(16)[y];bits=np.vstack([core.CURRENT,*core.AFTER]).T
    marginal=(p@bits-bits[y])**2
    return {'joint_log_loss':-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1.)),
        'joint_Brier':((p-target)**2).sum(1),
        **{name:marginal[:,i] for i,name in enumerate(('current_Brier','repair_Brier','retry_Brier','Complete_Brier'))},
        'mean_four_marginal_Brier':marginal.mean(1)}


def diagnostics(data,predictions):
    rows=data['records']; scores={name:losses(data['labels'],p) for name,p in predictions.items()}
    panels={}
    for subject in ('all','chemistry','physics'):
        for path in ('all','','repair','retry'):
            mask=np.asarray([(subject=='all' or r['subject']==subject) and (path=='all' or r['path']==path) for r in rows])
            panels[subject+'/'+(path or 'root')]={'nodes':int(mask.sum()), 'variants':{
                name:{metric:float(values[mask].mean()) for metric,values in loss.items()} for name,loss in scores.items()}}
    ids=sorted({r['id'] for r in rows}); groups=[[i for i,r in enumerate(rows) if r['id']==qid] for qid in ids]
    if len(ids)!=256 or any(len(g)!=3 for g in groups): raise ValueError('Paired inference requires256 complete question groups')
    draws=np.random.default_rng(SEED).integers(0,len(ids),size=(BOOTSTRAPS,len(ids)))
    comparisons={}
    for metric in scores[VARIANTS[0]]:
        delta=scores[VARIANTS[1]][metric]-scores[VARIANTS[0]][metric]
        per_question=np.asarray([delta[g].mean() for g in groups]); boot=per_question[draws].mean(1)
        comparisons[metric]={'V4_minus_old':float(per_question.mean()),'paired_question_CI95':np.quantile(boot,[.025,.975]).tolist()}
    screen=all(comparisons[m]['V4_minus_old']<0 for m in ('joint_log_loss','mean_four_marginal_Brier'))
    return {'panels':panels,'comparisons':comparisons,'screen_passed':screen,'nested_crossfit':False,
        'crossfit_limitation':EXCLUSION_NOTE,'controller_accuracy_claim':False,'fresh_validation':False}


def run(args):
    with training_only_reads(),guarded.no_model_calls(),threadpool_limits(limits=1):
        data,receipt=load_inputs(args); spec=specification(receipt)
        if args.mode=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',spec)
            print({'registered':True,'registration_sha256':sha(args.output/'registration.json'),'scored':False},flush=True);return
        if (not args.registration_sha256 or sha(args.output/'registration.json')!=args.registration_sha256
                or io.read(args.output/'registration.json')!=spec):raise ValueError('Changed explicit diagnostic registration')
        with io.lock(args.output/'.diagnostic.lock'):
            if any(p.name not in ('registration.json','.diagnostic.lock') for p in args.output.iterdir()):
                raise ValueError('Preserve earlier or partial diagnostic outputs')
            predictions,receipts={},{}
            for name in VARIANTS:
                predictions[name]=np.full((len(data['labels']),16),np.nan);receipts[name]=[]
                for fold in range(3):
                    held,p,model,record=outer_prediction(data,name,fold);predictions[name][held]=p
                    path=args.output/(name+'-fold'+str(fold)+'.pkl');io.atomic_bytes(path,pickle.dumps(model,protocol=5))
                    receipts[name].append({**record,'model_sha256':sha(path)})
                if not np.isfinite(predictions[name]).all():raise ValueError('Incomplete held-question predictions')
            report=diagnostics(data,predictions)
            io.npz(args.output/'private-oof.npz',predictions);io.write(args.output/'fold-receipts.json',receipts)
            for path,digest in {**spec['source_files'],**spec['input_evidence']}.items():
                if sha(path)!=digest:raise ValueError('Evidence changed during CPU-only diagnostic')
            io.write(args.output/'summary.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
                'training_questions':256,'decision_nodes':768,'new_neural_calls':0,'new_cloud_calls':0,
                'development_or_test_read':False,'reports':report,'fold_receipts':receipts,
                'source_sha256':sha(__file__)})
            io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
                'artifacts':{p.name:sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')},
                'new_neural_calls':0,'new_cloud_calls':0,'development_or_test_read':False})
            print({'complete':True,'screen_passed':report['screen_passed'],'comparisons':report['comparisons']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('register','diagnose'))
    for name in ('base-fit','verifier','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registration-sha256');run(parser.parse_args())
