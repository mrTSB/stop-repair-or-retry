"""Select deeper-review hybrids on retired development questions only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
from itertools import product
import json
from pathlib import Path
import pickle
import sys
import threading

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from semantic_data import read
from study_common import statistics, sha
from difficulty_pomdp import MissingObservation
from reasoning_pilot import short_repair
from long_review_pilot import review
from medium_late_pairs import APP, PROTOCOL, scoped
from banks import ordinary, committed, variant_observations, medium_view, canonical_review
from transfer_variants import actor
from transport import resilient
import extended_acquire as old
import medium_pomdp as semantic
import dynamic_grounded_pomdp as perception


def run(args):
    train, tt, tj, _ = ordinary(args.data, 'train')
    dev, dt, dj, _ = ordinary(args.data, 'dev')
    observations = variant_observations(args.medium, args.variants, 'medium-short')
    tt, tj = medium_view(train, tt, tj, observations, validate=True)
    metadata = read(args.examples / 'train.jsonl')
    training = np.load(args.verifier / 'oof.npz')
    supplemental = read(args.supplement / 'metadata.jsonl')
    predictions = np.load(args.supplement / 'oof.npz')
    receipt = json.loads((args.supplement / 'oof.receipt.json').read_bytes())
    if receipt['inputs_sha256'] != sha(args.supplement / 'inputs.jsonl') or receipt['references_present']:
        raise ValueError('Unverified supplemental predictions')
    metadata, outputs = semantic.merged_observations(metadata, training, supplemental, predictions)
    if not {r['id'] for r in metadata} <= {r['id'] for r in train}:
        raise ValueError('Non-training question entered a model fit')
    if {r['id'] for r in train} & {r['id'] for r in dev}:
        raise ValueError('Development/training overlap')
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {str(p): sha(p) for p in (
        args.data / 'train-observations.jsonl', args.data / 'train-late-observations.jsonl',
        args.examples / 'train.jsonl', args.verifier / 'contract.json', args.verifier / 'oof.npz',
        args.medium / 'train.jsonl', args.variants / 'train.jsonl',
        args.supplement / 'inputs.jsonl', args.supplement / 'metadata.jsonl',
        args.supplement / 'oof.npz', args.supplement / 'oof.receipt.json')}
    contract = args.output / 'identity.json'
    if contract.exists() and json.loads(contract.read_bytes()) != identity:
        raise ValueError('Changed development inputs; choose a new output directory')
    contract.write_text(json.dumps(identity, indent=2) + '\n')
    bank = args.medium / 'dev.jsonl'
    dt, dj = medium_view(dev, dt, dj, committed(bank), validate=True)
    summary = json.loads((args.verifier / 'training-summary.json').read_bytes())
    observer = perception.Observer(str((args.verifier / 'full').resolve()),
                                   summary['temperature_selected_on_training_oof'], device=args.device)
    observer.load()
    cache_path = args.output / 'runtime-cache.pkl'
    if cache_path.exists():
        observer._cache.update(pickle.loads(cache_path.read_bytes()))
    request = None
    if args.collect:
        import modal
        request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
        tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
        assert tags[old.MODEL] == old.MODEL_DIGEST and tags[old.JUDGE] == old.JUDGE_DIGEST
    bank.parent.mkdir(parents=True, exist_ok=True)
    locks = {'write': threading.Lock()}
    new_nodes = 0
    with bank.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def acquire(row, missing):
            nonlocal new_nodes
            if request is None:
                raise RuntimeError('Missing development response; use --collect') from missing
            qid, path, kind = row['id'], missing.path, missing.kind
            tree = dt[qid]
            safe = {**row, 'answer': ''}
            if kind == 'judge':
                if path not in ('repair', 'retry', 'guided') or path not in tree:
                    raise ValueError('Ordinary development response must exist before a deeper review')
                node = review(request, scoped(safe, path), tree[path], 1024, 512)
                node['protocol'] = PROTOCOL
                node['backend'] = node['calls'][0]['backend']
                dj[qid, path] = canonical_review(node)
            elif path.endswith('/reviewed'):
                parent = path.rsplit('/', 1)[0]
                if (qid, parent) not in dj:
                    acquire(row, MissingObservation('judge', parent))
                prompt = short_repair(safe, tree[parent], dj[qid, parent])
                seed = int(hashlib.sha256(f'{old.SEED}:{qid}:{parent}/repair'.encode()).hexdigest()[:8], 16) % 2**31
                node = actor(request, safe, prompt, seed, 128)
                node['protocol'] = 'medium-short-late-v1'
                tree[path] = node
            else:
                # Plain paths retain the exact original prompts, caps and seeds.
                if path not in old.PATHS or kind != 'solver':
                    raise ValueError('Unexpected ordinary path')
                old.request = request
                node = old.call(safe, path, tree, dj, False)
                node['protocol'] = 'ordinary-shared-v1'
                tree[path] = node
            if node['backend']['app'] != APP:
                raise ValueError('Unexpected solver backend')
            with locks['write']:
                stream.write(json.dumps({'id': qid, 'kind': kind, 'path': path, 'response': node}) + '\n')
                stream.flush()
                new_nodes += 1
                if new_nodes % 12 == 0:
                    print(f'Acquired {new_nodes} deeper-review development responses', flush=True)

        def batch(model, settings):
            def one(row):
                while True:
                    try:
                        result = semantic.evaluate(row, dt[row['id']], dj, model, settings, 'C')
                        break
                    except MissingObservation as missing:
                        acquire(row, missing)
                return result
            with ThreadPoolExecutor(max_workers=2) as pool:
                return list(pool.map(one, dev))

        candidates = []
        for states, smoothing, sem_review in product(args.states, args.smoothing, (False, True)):
            key = f'states{states}-smooth{smoothing:g}-review{int(sem_review)}'
            path = args.output / (key + '.pkl')
            if path.exists():
                model = pickle.loads(path.read_bytes())
            else:
                print('Fitting', key, flush=True)
                model = semantic.fit(train, tt, tj, args.verifier / 'full', metadata, outputs,
                                     smoothing=smoothing, semantic_review=sem_review, state_count=states)
                path.write_bytes(pickle.dumps(model, protocol=5))
            for name in ('_model', '_tokenizer', '_recipe', '_cache', 'device', '_retriever', '_backgrounds'):
                setattr(model.classifier, name, getattr(observer, name))
            model.review_observer = model.classifier if sem_review else None
            for penalty, threshold, future in product((0., .02, .04, .08, .12), (0., .02, .06), (False, True)):
                settings = {'context_weight': 0., 'penalty': penalty, 'science_weight': 1.,
                            'reuse_review': True, 'late_feedback': True, 'plan_reviews': future,
                            'continuous_filter': False, 'threshold': threshold}
                items = batch(model, settings)
                candidates.append({'family': 'C', 'module': 'medium_pomdp', 'model': key,
                                   'settings': settings, 'late': True, 'dev': statistics(items)})
                for d in model.dynamics.values():
                    d._values.clear(); d._information.clear(); d._alphas.clear()
            (args.output / 'search.json').write_text(json.dumps(candidates, indent=2) + '\n')
            cache_path.write_bytes(pickle.dumps(observer._cache, protocol=5))
            queried = [r for r in candidates if r['dev']['mean_judge_calls'] > 0]
            best = max(queried, key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute']))
            print('Completed', key, 'best', json.dumps(best), flush=True)
        reference = json.loads((args.reference / 'summary.json').read_bytes())
        b_cost = reference['selected']['B']['dev']['mean_compute']
        queried = [r for r in candidates if r['dev']['mean_judge_calls'] > 0]
        cheap = [r for r in queried if r['dev']['mean_compute'] < b_cost]
        result = {'best': max(queried, key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute'])),
                  'best_cheaper': max(cheap, key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute'])) if cheap else None,
                  'ordinary_reference': reference['selected'], 'fresh_data_used': False,
                  'training_questions': len(train), 'paired_review_questions': 192,
                  'development_questions': len(dev), 'new_development_nodes': new_nodes}
        (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'examples', 'verifier', 'medium', 'variants', 'supplement', 'output', 'reference'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--collect', action='store_true')
    p.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    p.add_argument('--states', type=int, nargs='+', choices=(2, 4), default=[4, 2])
    p.add_argument('--smoothing', type=float, nargs='+', default=[32.])
    run(p.parse_args())
