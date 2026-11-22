"""Independent TRAIN math screen of the same frozen reward observations."""
import argparse
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

import math_specialization_diagnostic as math
import reward_outcome_diagnostic as science

io, fixed, nested, observer = science.io, science.fixed, science.nested, science.observer
SCHEMA = 'frozen-reward-math256-conditional-TRAIN-v1'


def collapse(probability):
    if np.asarray(probability).shape[1] != 16: raise ValueError('Require registered joint representation')
    return np.asarray(probability).reshape(-1, 8, 2).sum(2)


def inputs(args):
    spec = io.read(args.math_diagnostic/'registration.json'); done = io.read(args.math_diagnostic/'complete.json')
    old_args = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()})
    all_data, expected = math.inputs(old_args)
    if expected != spec or done['registration_sha256'] != io.sha(args.math_diagnostic/'registration.json') or not done['screen_passed']:
        raise ValueError('Require the sealed retained math-specialized control')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    io.verify({str(args.math_diagnostic/k): v for k, v in done['digests'].items()})
    base_spec = io.read(old_args.base_fit/'registration.json')
    rows, trees = math.source.plain_data(Path(base_spec['paths']['data']), 'train')
    rows = [r for r in rows if r['subject'] == 'math']; trees = {r['id']: trees[r['id']] for r in rows}
    eligible = np.asarray([r['subject'] == 'math' for r in all_data['records']]); indices = np.flatnonzero(eligible)
    records = [r for r, keep in zip(all_data['records'], eligible) if keep]
    data = {'records': records, 'labels': all_data['y'][eligible]*2, 'valid': all_data['valid'][eligible]}
    base = all_data['x'][eligible, 20:].copy()
    if len(rows) != 256 or base.shape != (768, 110): raise ValueError('Changed full math cohort')
    with np.load(args.math_diagnostic/'private-oof.npz', allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['indices'], indices); control = archive['specialized'].copy()
    base_dir = Path(base_spec['paths']['base'])
    old = math.source.cached.observer_from(base_dir)
    by_id = {r['id']: r for r in rows}
    # Original training logits omitted token counts. Re-tokenize the verified
    # paid input locally, without an encoder pass or a max-length cost estimate.
    tokenizer = AutoTokenizer.from_pretrained(old.directory, local_files_only=True, trust_remote_code=False, use_fast=False)
    entries = [old.entry({**by_id[r['id']], 'answer': ''}, r['path'],
        {p: trees[r['id']][p] for p in ('', r['path'])}) for r in records]
    encoded = science.conditional.network.initial.encode(tokenizer, entries, old._recipe['max_length'])
    old_slots = {(r['id'], r['path']): int(n) for r, n in zip(records, encoded['attention_mask'].sum(1))}
    conversations = [observer.conversation({**by_id[r['id']], 'answer': ''}, r['path'],
        {p: trees[r['id']][p] for p in ('', r['path'])}) for r in records]
    assets = io.read(args.assets/'receipt.json')
    if assets['repository'] != observer.REPOSITORY or assets['revision'] != observer.REVISION or assets['remote_code']:
        raise ValueError('Changed frozen model')
    files = {str((args.assets/'model'/k).resolve()): v for k, v in assets['files'].items()}; io.verify(files)
    evidence = {**spec['input_evidence'], **files,
        **{str((args.math_diagnostic/k).resolve()): v for k, v in done['digests'].items()},
        str((args.assets/'receipt.json').resolve()): io.sha(args.assets/'receipt.json'),
        str((args.math_diagnostic/'complete.json').resolve()): io.sha(args.math_diagnostic/'complete.json')}
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]), 'input_evidence': evidence,
        'paths': {k: str(getattr(args, k).resolve()) for k in ('math_diagnostic', 'assets', 'output')},
        'model': assets, 'device': args.device, 'dtype': 'float32', 'batch_size': 4, 'max_length': observer.MAX_LENGTH,
        'overlength': 'First768 plus last768; every truncation reported',
        'training_questions': 256, 'observations': 768, 'settings': list(observer.SETTINGS),
        'selection': 'Three inner whole-question folds, minimum joint loss; three outer folds',
        'state_encoding': 'Eight math states encoded as even sixteen-state labels then adjacent probabilities summed; no Complete action',
        'old_neural_features_removed': 20, 'current_only_reward_input': True,
        'conversations_sha256': fixed.fingerprint(conversations), 'base_sha256': fixed.array_sha(base),
        'records_sha256': fixed.fingerprint(records), 'labels_sha256': fixed.array_sha(data['labels']),
        'screen': {'conditional_accuracy_gain_at_least': .02, 'compute_ratio_at_most': 1.10},
        'control': 'Retained nested math-specific outcome forecasts on the same256 questions',
        'compute': 'Paid prefix plus chosen terminal solver plus one current scorer, actual parameters and nonpadding input tokens /3.09',
        'control_token_slots': 'Retokenized exact verified old paid inputs; no old neural calls',
        'control_token_slots_sha256': fixed.fingerprint(sorted((q, p, n) for (q, p), n in old_slots.items())),
        'penalty': .02, 'cloud_calls': 0, 'solver_unchanged': True, 'development_or_test_read': False,
        'full_controller_claim': False, 'fresh_validation': False, 'pretraining_overlap_not_independently_established': True}
    return rows, trees, data, base, conversations, old_slots, control, old._recipe['parameters'], record


def conditional(rows, trees, data, probability, slots, parameters):
    core = math.core; by_id = {r['id']: r for r in rows}; costs = {}
    for fold in range(3):
        sums, counts = np.zeros((2, 8)), np.zeros((2, 8))
        for r in data['records']:
            if not r['path'] or r['fold'] == fold: continue
            tree = trees[r['id']]; path = r['path']; truth = by_id[r['id']]['answer']
            state = core.state(*[core.correct(tree[p], truth) for p in (path, path+'/repair', path+'/retry')])
            for a, edge in enumerate(('repair', 'retry')):
                node = tree[path+'/'+edge]; sums[a, state] += 4*(node['input_tokens']+node['output_tokens'])/3.09; counts[a, state] += 1
        pooled = sums.sum(1)/np.maximum(counts.sum(1), 1)
        costs[fold] = (sums+4*pooled[:, None])/(counts+4)
    items = []
    for i, r in enumerate(data['records']):
        qid, path = r['id'], r['path']
        if not path: continue
        values = [probability[i]@core.CURRENT, *[probability[i]@(core.AFTER[a]-.02*costs[r['fold']][a]/1000) for a in range(2)]]
        action = int(np.argmax(values)); final = path if action == 0 else path+'/'+('repair', 'retry')[action-1]
        visited = ['', path]+([] if action == 0 else [final]); tree = trees[qid]
        tokens = sum(tree[p]['input_tokens']+tree[p]['output_tokens'] for p in visited)
        items.append({'id': qid, 'path': path, 'action': action, 'correct': core.correct(tree[final], by_id[qid]['answer']),
            'compute': (4*tokens+parameters/1e9*slots[qid, path])/3.09})
    return items


def diagnose(args, rows, trees, data, base, conversations, slots, control, old_parameters, record):
    folder = args.output/'features'; done = io.read(folder/'complete.json')
    if done['registration_sha256'] != args.registration_sha256: raise ValueError('Changed observation seal')
    io.verify({str(folder/k): v for k, v in done['digests'].items()}); metadata = io.read(folder/'metadata.json')
    with np.load(folder/'all.npz', allow_pickle=False) as archive:
        scores, hidden, tokens = (archive[k].copy() for k in ('scores', 'hidden', 'tokens'))
    folds = np.asarray([r['fold'] for r in data['records']]); predicted = np.full((len(base), 8), np.nan); receipts = []
    for fold in range(3):
        fit, held = folds != fold, folds == fold; receipt = nested.group_receipt(data['records'], fit, held)
        setting, search = science.choose(data, base, scores, hidden, fit)
        model = observer.fit(base[fit], scores[fit], hidden[fit], data['labels'][fit], setting)
        predicted[held] = collapse(model.predict(base[held], scores[held], hidden[held], data['valid'][held]))
        io.atomic_bytes(args.output/f'outer{fold}.pkl', pickle.dumps(model, protocol=5))
        receipts.append({**receipt, 'fold': fold, 'setting': setting, 'inner_search': search})
        print({'fold': fold, 'selected': setting}, flush=True)
    if not np.isfinite(predicted).all(): raise ValueError('Incomplete whole-question predictions')
    new_slots = {(r['id'], r['path']): int(n) for r, n in zip(data['records'], tokens)}
    before = conditional(rows, trees, data, control, slots, old_parameters)
    after = conditional(rows, trees, data, predicted, new_slots, metadata['parameters'])
    metrics = lambda v: {'n': len(v), 'correct': sum(int(r['correct']) for r in v),
        'accuracy': float(np.mean([r['correct'] for r in v])), 'mean_compute': float(np.mean([r['compute'] for r in v]))}
    a, b = metrics(before), metrics(after); delta = b['accuracy']-a['accuracy']; ratio = b['mean_compute']/a['mean_compute']
    groups = {r['id']: [] for r in before}
    for i, r in enumerate(before): groups[r['id']].append(i)
    if len(groups) != 256 or any(len(v) != 2 for v in groups.values()): raise ValueError('Changed paired cohort')
    diff = np.asarray([np.mean([int(after[i]['correct'])-int(before[i]['correct']) for i in g]) for g in groups.values()])
    draws = np.random.default_rng(238031).integers(0, 256, (2000, 256))
    screen = {'accuracy_gain': delta, 'compute_ratio': ratio, 'passed': delta >= .02 and ratio <= 1.10,
        'paired_question_accuracy_CI95': np.quantile(diff[draws].mean(1), [.025, .975]).tolist()}
    result = {'schema': SCHEMA, 'control': a, 'candidate': b, 'screen': screen, 'features': metadata,
        'forecast_losses': {name: {k: float(v.mean()) for k, v in math.losses(data['labels']//2, p).items()}
            for name, p in (('control', control), ('candidate', predicted))},
        'full_controller_claim': False, 'fresh_validation': False, 'development_or_test_read': False, 'cloud_calls': 0}
    io.npz(args.output/'oof.npz', {'probability': predicted}); io.write(args.output/'fold-receipts.json', receipts)
    io.write(args.output/'summary.json', result); io.write(args.output/'private-conditional-items.json', {'control': before, 'candidate': after})
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'screen_passed': screen['passed'], 'digests': {str(p.relative_to(args.output)): io.sha(p) for p in args.output.rglob('*')
            if p.is_file() and p != args.output/'complete.json' and not p.name.startswith('.')}})
    print({'control': a, 'candidate': b, 'screen': screen}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=('register', 'collect', 'diagnose'))
    for name in ('math-diagnostic', 'assets', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps'); parser.add_argument('--registration-sha256'); args = parser.parse_args()
    with fixed.training_only_reads(), threadpool_limits(limits=1):
        with fixed.guarded.no_model_calls(): bundle = inputs(args)
        record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}, flush=True); return
        if not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256 or io.read(args.output/'registration.json') != record:
            raise ValueError('Changed registered math experiment')
        with io.lock(args.output/'.run.lock'):
            if (args.output/'complete.json').exists(): raise ValueError('Preserve completed experiment')
            if args.phase == 'collect': science.collect(args, bundle[4], record)
            else:
                if any(args.output.glob('outer*.pkl')): raise ValueError('Preserve partial fitting')
                with fixed.guarded.no_model_calls(): diagnose(args, *bundle)


if __name__ == '__main__': main()
