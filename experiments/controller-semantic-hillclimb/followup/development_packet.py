"""Train and select the one-brief-review, full-packet hybrid on retired data."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
from itertools import product
import json
from pathlib import Path
import pickle
import threading

import numpy as np

from banks import ordinary, committed
from full_packet_pilot import prompt, PROTOCOL, sha
from packet_bank import PLAIN_PROTOCOL
from reasoning_pilot import inference
from difficulty_pomdp import MissingObservation
from semantic_data import read
from study_common import statistics
from transport import resilient
from medium_late_pairs import APP
import dynamic_grounded_pomdp as perception
import packet_pomdp as semantic
import extended_acquire as old
from transfer_variants import actor


def run(args):
    train, tt, original_train, _ = ordinary(args.data, 'train')
    dev, dt, original_dev, _ = ordinary(args.data, 'dev')
    tt, tj = semantic.view(train, tt, original_train, committed(args.packet / 'train.jsonl'), validate=True)
    metadata, outputs = semantic.training_observations(
        read(args.examples / 'train.jsonl'), np.load(args.verifier / 'oof.npz'),
        read(args.supplement / 'metadata.jsonl'), np.load(args.supplement / 'oof.npz'))
    receipt = json.loads((args.supplement / 'oof.receipt.json').read_bytes())
    if receipt['inputs_sha256'] != sha(args.supplement / 'inputs.jsonl') or receipt['references_present']:
        raise ValueError('Unverified local observations')
    if not {r['id'] for r in metadata} <= {r['id'] for r in train}:
        raise ValueError('Non-training question entered fit')
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {str(p): sha(p) for p in (
        args.data / 'train-observations.jsonl', args.data / 'train-late-observations.jsonl',
        args.examples / 'train.jsonl', args.verifier / 'contract.json', args.verifier / 'oof.npz',
        args.packet / 'contract.json', args.packet / 'train.jsonl', args.supplement / 'manifest.json',
        args.supplement / 'inputs.jsonl', args.supplement / 'metadata.jsonl',
        args.supplement / 'oof.npz', args.supplement / 'oof.receipt.json')}
    identity['semantic_code'] = sha(semantic.__file__)
    contract = args.output / 'identity.json'
    if contract.exists() and json.loads(contract.read_bytes()) != identity:
        raise ValueError('Changed development inputs')
    contract.write_text(json.dumps(identity, indent=2) + '\n')
    bank = args.packet / 'dev.jsonl'
    dt, dj = semantic.view(dev, dt, original_dev, committed(bank), validate=True)
    summary = json.loads((args.verifier / 'training-summary.json').read_bytes())
    observer = perception.Observer(str((args.verifier / 'full').resolve()),
                                   summary['temperature_selected_on_training_oof'], device=args.device)
    observer.load()
    cache = args.output / 'runtime-cache.pkl'
    if cache.exists():
        observer._cache.update(pickle.loads(cache.read_bytes()))
    request = None
    if args.collect:
        import modal
        request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
        tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
        assert tags[old.MODEL] == old.MODEL_DIGEST and tags[old.JUDGE] == old.JUDGE_DIGEST
        old.request = request
    lock = threading.Lock()
    new_nodes = 0
    with bank.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def acquire(row, missing):
            nonlocal new_nodes
            qid, kind, path = row['id'], missing.kind, missing.path
            safe = {**row, 'answer': ''}; tree = dt[qid]
            if kind == 'judge' and path == '':
                # All ordinary development root reviews are already available.
                # Copy exact receipts; do not regenerate or charge a new API call.
                node = original_dev[qid, '']
                dj[qid, ''] = node
            else:
                if request is None:
                    raise RuntimeError('Missing compact response; use --collect') from missing
                parent = path.rsplit('/', 1)[0] if '/' in path else ''
                if parent not in tree:
                    acquire(row, MissingObservation('solver', parent))
                if path == 'guided' or path.endswith('/reuse'):
                    if (qid, '') not in dj:
                        acquire(row, MissingObservation('judge', ''))
                    text = prompt(safe, tree[parent], dj[qid, ''])
                    logical = path.replace('guided', 'repair').replace('reuse', 'repair')
                    seed = int(hashlib.sha256(f'{old.SEED}:{qid}:{logical}'.encode()).hexdigest()[:8], 16) % 2**31
                    node = actor(request, safe, text, seed, 128)
                    node['protocol'] = PROTOCOL
                elif path == 'guided/retry':
                    node = old.call(safe, path, tree, dj, False)
                    node['protocol'] = PLAIN_PROTOCOL
                else:
                    if kind != 'solver' or path not in old.PATHS or path.startswith('guided'):
                        raise ValueError('Unexpected packet-hybrid path')
                    node = old.call(safe, path, tree, dj, False)
                    node['protocol'] = 'ordinary-shared-v1'
                if node['backend']['app'] != APP:
                    raise ValueError('Unexpected acquisition backend')
                tree[path] = node
            with lock:
                stream.write(json.dumps({'id': qid, 'kind': kind, 'path': path, 'response': node}) + '\n')
                stream.flush()
                new_nodes += 1
                if new_nodes % 24 == 0:
                    print(f'Acquired {new_nodes} full-packet development records', flush=True)

        def batch(model, settings):
            def one(row):
                while True:
                    try:
                        return semantic.evaluate(row, dt[row['id']], dj, model, settings, 'C')
                    except MissingObservation as missing:
                        acquire(row, missing)
            with ThreadPoolExecutor(max_workers=2) as pool:
                return list(pool.map(one, dev))

        candidates = []
        for smoothing, sem_review in product(args.smoothing, (False, True)):
            key = f'smooth{smoothing:g}-review{int(sem_review)}'
            path = args.output / (key + '.pkl')
            if path.exists():
                model = pickle.loads(path.read_bytes())
            else:
                print('Fitting full-packet hybrid', key, flush=True)
                model = semantic.fit(train, tt, tj, args.verifier / 'full', metadata, outputs,
                                     smoothing=smoothing, semantic_review=sem_review)
                path.write_bytes(pickle.dumps(model, protocol=5))
            for name in ('_model', '_tokenizer', '_recipe', '_cache', 'device', '_retriever', '_backgrounds'):
                setattr(model.classifier, name, getattr(observer, name))
            model.review_observer = model.classifier if sem_review else None
            for penalty, threshold, continuous in product((.02, .04, .08, .12, .18, 0.), (0., .03, .08), (False, True)):
                settings = {'context_weight': 0., 'penalty': penalty, 'science_weight': 1.,
                            'reuse_review': True, 'plan_reviews': False,
                            'continuous_root': continuous, 'threshold': threshold}
                items = batch(model, settings)
                candidates.append({'family': 'C', 'module': 'packet_pomdp', 'model': key,
                                   'settings': settings, 'late': False, 'dev': statistics(items)})
                for d in model.dynamics.values():
                    d._alphas.clear()
            (args.output / 'search.json').write_text(json.dumps(candidates, indent=2) + '\n')
            cache.write_bytes(pickle.dumps(observer._cache, protocol=5))
            print('Completed full-packet hybrid', key, flush=True)
        reference = json.loads((args.reference / 'summary.json').read_bytes())
        b_cost = reference['selected']['B']['dev']['mean_compute']
        queried = [r for r in candidates if r['dev']['mean_judge_calls'] > 0]
        cheap = [r for r in queried if r['dev']['mean_compute'] < b_cost]
        best = lambda group: max(group, key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute']))
        result = {'best': best(queried), 'best_cheaper': best(cheap) if cheap else None,
                  'ordinary_reference': reference['selected'], 'fresh_data_used': False,
                  'training_questions': len(train), 'paired_review_questions': 192,
                  'development_questions': len(dev), 'new_development_nodes': new_nodes,
                  'new_judge_api_calls': 0, 'root_review': 'Exact original brief review, reused at full per-policy cost'}
        (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'examples', 'verifier', 'packet', 'supplement', 'output', 'reference'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    p.add_argument('--smoothing', type=float, nargs='+', default=[4., 16., 32.])
    p.add_argument('--collect', action='store_true')
    run(p.parse_args())
