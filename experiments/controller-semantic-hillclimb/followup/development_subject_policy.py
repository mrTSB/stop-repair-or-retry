"""Allocate the POMDP inference budget by observed subject on retired development data."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
import importlib
from itertools import product
import json
from pathlib import Path
import pickle
import shutil
import threading

from banks import ordinary, committed
from confirmation_common import sha, clear
import subject_policy_pomdp as semantic
from study_common import statistics, BASELINES
from study import fixed
from late_pairs import llm_policy, judge_prompt, sanitized
from review_pilot import inference
from difficulty_pomdp import MissingObservation
from transport import resilient
import extended_acquire as old


def pareto(options):
    """Keep only actual subject policies not dominated in accuracy and cost."""
    best_at_correct = {}
    for entry in options:
        key = entry['metrics']['correct']
        if key not in best_at_correct or entry['metrics']['mean_compute'] < best_at_correct[key]['metrics']['mean_compute']:
            best_at_correct[key] = entry
    answer = []
    cheapest = float('inf')
    for _, entry in sorted(best_at_correct.items(), reverse=True):
        if entry['metrics']['mean_compute'] < cheapest:
            answer.append(entry)
            cheapest = entry['metrics']['mean_compute']
    return answer


def run(args):
    import modal
    semantic = importlib.import_module('subject_packet_pomdp' if args.packet else 'subject_policy_pomdp')
    rows, trees, original, late = ordinary(args.data, 'dev')
    original_trees = trees
    if args.packet:
        trees, late = semantic.view(rows, trees, original, committed(args.packet / 'dev.jsonl'), validate=True)
    if len(rows) != 240:
        raise ValueError('Only the retired240development questions may select policies')
    reference = json.loads((args.probe / 'summary.json').read_bytes())
    strong_b = (reference['selected'] if 'selected' in reference else reference['ordinary_reference'])['B']
    args.output.mkdir(parents=True, exist_ok=True)
    model_paths = sorted(args.probe.glob('smooth32-review*.pkl' if args.packet else 'smooth32-review*-late1.pkl'))
    if not model_paths:
        raise ValueError('No completed training models in the requested development probe')
    identity = {'semantic_code': sha(semantic.__file__), 'probe': str(args.probe),
                'model_sha256': {p.name: sha(p) for p in model_paths},
                'train_sha256': sha(args.data / 'train-observations.jsonl'),
                'selection': 'One fixed settings tuple per observed subject, one observer/planner per controller; joint C compute below strongest ordinary B. No test outcomes.'}
    identity['future_review_flags'] = args.future_reviews
    identity['families'] = args.families
    identity['context_weights'] = args.context_weights
    if args.packet:
        identity['packet_training_sha256'] = sha(args.packet / 'train.jsonl')
    receipt = args.output / 'identity.json'
    if receipt.exists() and json.loads(receipt.read_bytes()) != identity:
        raise ValueError('Changed development search; choose a new output')
    receipt.write_text(json.dumps(identity, indent=2) + '\n')
    request = resilient(modal.Cls.from_name('srr-followup-solver-20261120', 'Ollama')(),
                        args.data / 'development-transport-errors.jsonl')
    tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    assert tags[old.MODEL] == old.MODEL_DIGEST and tags[old.JUDGE] == old.JUDGE_DIGEST
    old.request = request
    lock = threading.Lock()
    acquired = 0
    shared = None
    candidates = []
    with ExitStack() as stack:
        base_stream = stack.enter_context((args.data / 'dev-observations.jsonl').open('a'))
        late_stream = stack.enter_context((args.data / 'dev-late-observations.jsonl').open('a'))
        packet_stream = stack.enter_context((args.packet / 'dev.jsonl').open('a')) if args.packet else None
        for stream in ((packet_stream,) if args.packet else (base_stream, late_stream)):
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def acquire(row, kind, path):
            nonlocal acquired
            tree = trees[row['id']]
            safe = {**row, 'answer': ''}
            if args.packet:
                from full_packet_pilot import prompt, PROTOCOL
                from packet_bank import PLAIN_PROTOCOL
                from transfer_variants import actor
                if kind == 'judge' and not path:
                    node = original[row['id'], '']
                    late[row['id'], ''] = node
                else:
                    parent = path.rsplit('/', 1)[0] if '/' in path else ''
                    if parent not in tree:
                        acquire(row, 'solver', parent)
                    if path == 'guided' or path.endswith('/reuse'):
                        if (row['id'], '') not in late:
                            acquire(row, 'judge', '')
                        text = prompt(safe, tree[parent], late[row['id'], ''])
                        logical = path.replace('guided', 'repair').replace('reuse', 'repair')
                        seed = int(hashlib.sha256(f'{old.SEED}:{row["id"]}:{logical}'.encode()).hexdigest()[:8], 16) % 2**31
                        node = actor(request, safe, text, seed, 128)
                        node['protocol'] = PROTOCOL
                    else:
                        if kind != 'solver' or path not in old.PATHS:
                            raise ValueError('Unexpected packet continuation')
                        node = old.call(safe, path, tree, late, False)
                        node['protocol'] = PLAIN_PROTOCOL if path == 'guided/retry' else 'ordinary-shared-v1'
                    tree[path] = node
                with lock:
                    packet_stream.write(json.dumps({'id': row['id'], 'kind': kind, 'path': path, 'response': node}) + '\n')
                    packet_stream.flush()
                    acquired += 1
                return
            if kind == 'judge' and path not in tree:
                acquire(row, 'solver', path)
            if kind == 'solver' and path:
                parent = path.rsplit('/', 1)[0] if '/' in path else ''
                if parent not in tree:
                    acquire(row, 'solver', parent)
                if path == 'guided' and (row['id'], '') not in original:
                    acquire(row, 'judge', '')
                if path.endswith('/reviewed') and (row['id'], parent) not in late:
                    acquire(row, 'judge', parent)
            storage = base_stream
            if kind == 'judge' and path:
                node = sanitized(inference(request, safe, judge_prompt(safe, path, tree), 'brief', True, path))
                node['protocol'] = 'late-reviewed-repair-v1'
                late[row['id'], path] = node
                storage = late_stream
            elif kind == 'solver' and path.endswith('/reviewed'):
                parent = path.rsplit('/', 1)[0]
                text = old.repair_prompt(safe, tree[parent], old.feedback(late[row['id'], parent]))
                node = inference(request, safe, text, 'brief', False, parent + '/repair')
                node['protocol'] = 'late-reviewed-repair-v1'
                tree[path] = node
                storage = late_stream
            else:
                if (kind == 'solver' and path not in old.PATHS) or (kind == 'judge' and path not in old.CHECKPOINTS):
                    raise ValueError('Unexpected ordinary request')
                node = old.call(safe, path, tree, original, kind == 'judge')
                if kind == 'solver':
                    tree[path] = node
                else:
                    original[row['id'], path] = node
                    if not path:
                        late[row['id'], path] = node
            assert node['backend']['app'] == 'srr-followup-solver-20261120'
            with lock:
                storage.write(json.dumps({'id': row['id'], 'kind': kind, 'path': path, 'response': node}) + '\n')
                storage.flush()
                acquired += 1

        def batch(model, configuration, family):
            def one(row):
                while True:
                    try:
                        return semantic.evaluate(row, trees[row['id']], late, model, configuration, family)
                    except MissingObservation as missing:
                        acquire(row, missing.kind, missing.path)
            with ThreadPoolExecutor(max_workers=2) as pool:
                return list(pool.map(one, rows))

        for path in model_paths:
            model = pickle.loads(path.read_bytes())
            model.classifier.device = args.device
            if shared is None:
                model.classifier.load()
                shared = model.classifier
                cache = args.probe / 'runtime-cache.pkl'
                if cache.exists():
                    shared._cache.update(pickle.loads(cache.read_bytes()))
            else:
                for attr in ('_model', '_tokenizer', '_recipe', '_cache', 'device', '_retriever', '_backgrounds'):
                    setattr(model.classifier, attr, getattr(shared, attr))
            destination = args.output / path.name
            if not destination.exists():
                shutil.copy2(path, destination)
            families = ('C',) if args.packet else tuple(args.families)
            options = {family: {s: [] for s in ('math', 'chemistry', 'physics')} for family in families}
            for penalty, continuous, planned, weight in product((0., .01, .02, .04, .06, .08, .10, .16, .24),
                                                                 (False, True), [v == 'on' for v in args.future_reviews],
                                                                 args.context_weights):
                if planned and (continuous or args.packet):
                    # The exact future-review backup is implemented for the
                    # finite ordinary observation model, not these alternatives.
                    continue
                base = {'context_weight': weight, 'penalty': penalty, 'science_weight': 1.,
                        'reuse_review': True, 'late_feedback': True, 'plan_reviews': planned,
                        ('continuous_root' if args.packet else 'continuous_filter'): continuous}
                for family in families:
                    for threshold in ((0.,) if family == 'A' else (0., .03, .08)):
                        setting = {**base, **({'threshold': threshold} if family == 'C' else {})}
                        items = batch(model, {'by_subject': {s: setting for s in options[family]}}, family)
                        for subject in options[family]:
                            subset = [r for r in items if r['subject'] == subject]
                            options[family][subject].append({'settings': setting, 'metrics': statistics(subset), 'items': subset})
                for d in model.dynamics.values():
                    for name in ('_values', '_information', '_alphas'):
                        getattr(d, name, {}).clear()
            for family in families:
                fronts = [pareto(options[family][s]) for s in ('math', 'chemistry', 'physics')]
                choices = []
                for combination in product(*fronts):
                    items = [r for entry in combination for r in entry['items']]
                    measured = statistics(items)
                    if family == 'C' and (measured['mean_judge_calls'] <= 0 or measured['mean_compute'] >= strong_b['dev']['mean_compute']):
                        continue
                    choices.append((measured, combination))
                if not choices:
                    continue
                measured, combination = max(choices, key=lambda p: (p[0]['accuracy'], -p[0]['mean_compute']))
                candidates.append({'family': family, 'module': semantic.__name__, 'model': path.stem,
                                   'late': True, 'settings': {'by_subject': {s: entry['settings'] for s, entry in zip(('math', 'chemistry', 'physics'), combination)}},
                                   'dev': measured})
            (args.output / 'search.json').write_text(json.dumps(candidates, indent=2) + '\n')
            print('Completed subject budget allocation', path.stem, json.dumps(candidates[-2:]), flush=True)
    selected = {family: max((r for r in candidates if r['family'] == family), key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute'])) for family in families}
    selected['B'] = strong_b
    baselines = {name: statistics([fixed(r, original_trees[r['id']], *setting) for r in rows]) for name, setting in BASELINES.items()}
    summary = {'selected': selected, 'baselines': baselines, 'fresh_data_used': False,
               'training_questions': 1472, 'development_questions': 240, 'new_development_nodes': acquired,
               'group_policy_selection': 'Subject known before inference; a single fixed observer/planner and subject-specific cost penalties/thresholds. Global C compute constrained below strong ordinary B.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'probe', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    p.add_argument('--packet', type=Path)
    p.add_argument('--future-reviews', choices=('off', 'on'), nargs='+', default=['off'])
    p.add_argument('--families', choices=('A', 'C'), nargs='+', default=['A', 'C'])
    p.add_argument('--context-weights', type=float, nargs='+', default=[0.])
    run(p.parse_args())
