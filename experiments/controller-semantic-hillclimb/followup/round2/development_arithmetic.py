"""Test arithmetic observations on the retired development panel, preserving freezes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
from pathlib import Path
import pickle
import shutil
import sys
import threading

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import ordinary
from semantic_data import read
from confirmation_common import sha
from study_common import statistics, BASELINES
from study import fixed
from late_pairs import llm_policy
from difficulty_pomdp import MissingObservation
from transport import resilient
import extended_acquire as old
import arithmetic_pomdp as semantic


def run(args):
    import modal
    args.output.mkdir(parents=True, exist_ok=True)
    data = args.output / 'data'
    if not data.exists():
        (data / 'splits').mkdir(parents=True)
        shutil.copy2(args.source / 'splits/selection.json', data / 'splits/selection.json')
        for split in ('train', 'dev'):
            shutil.copy2(args.source / 'splits' / (split + '-questions.jsonl'), data / 'splits' / (split + '-questions.jsonl'))
            for suffix in ('observations', 'late-observations'):
                shutil.copy2(args.source / (split + '-' + suffix + '.jsonl'), data / (split + '-' + suffix + '.jsonl'))
    train, trees, _, reviews = ordinary(data, 'train')
    dev, dt, _, dj = ordinary(data, 'dev')
    metadata = read(args.examples)
    outputs = np.load(args.verifier / 'oof.npz')
    contract = json.loads((args.verifier / 'contract.json').read_bytes())
    if contract['data_sha256'] != sha(args.examples):
        raise ValueError('Changed original observer fitting inputs')
    if len(dev) != 240:
        raise ValueError('This pilot uses only the retired240, not the new test panel')
    identity = {'source_sha256': sha(__file__), 'arithmetic_pomdp_sha256': sha(semantic.__file__),
                'arithmetic_sha256': sha(Path(__file__).with_name('numeric_consistency.py')),
                'examples_sha256': sha(args.examples), 'oof_sha256': sha(args.verifier / 'oof.npz'),
                'source_train_sha256': sha(args.source / 'train-observations.jsonl'),
                'source_dev_sha256': sha(args.source / 'dev-observations.jsonl'),
                'role': 'Retired development pilot only. Original frozen banks read-only; new nodes go only to this copy.'}
    receipt = args.output / 'identity.json'
    if receipt.exists() and json.loads(receipt.read_bytes()) != identity:
        raise ValueError('Changed arithmetic development pilot')
    receipt.write_text(json.dumps(identity, indent=2) + '\n')
    model_file = args.output / 'model.pkl'
    if model_file.exists():
        model = pickle.loads(model_file.read_bytes())
    else:
        model = semantic.fit(train, trees, reviews, args.verifier / 'full', metadata, outputs,
                             smoothing=32., late=True, semantic_review=False)
        model_file.write_bytes(pickle.dumps(model, protocol=5))
    model.classifier.device = args.device
    model.classifier.load()
    request = resilient(modal.Cls.from_name('srr-followup-solver-20261120', 'Ollama')(), data / 'transport-errors.jsonl')
    old.request = request
    actual = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    assert actual[old.MODEL] == old.MODEL_DIGEST
    baseline = {name: statistics([fixed(r, dt[r['id']], *p) for r in dev]) for name, p in BASELINES.items()}
    b_settings = {'judge_stop': 'agreement', 'reuse_review': False}
    b = statistics([llm_policy(r, dt[r['id']], dj, b_settings) for r in dev])
    lock = threading.Lock()
    acquired = 0
    candidates = []
    with (data / 'dev-observations.jsonl').open('a') as stream:
        def one(row, settings):
            nonlocal acquired
            safe = {**row, 'answer': ''}
            while True:
                try:
                    return semantic.evaluate(row, dt[row['id']], dj, model, settings, 'A')
                except MissingObservation as missing:
                    if missing.kind != 'solver' or missing.path not in old.PATHS:
                        raise ValueError('A must not acquire an LLM review')
                    node = old.call(safe, missing.path, dt[row['id']], dj, False)
                    with lock:
                        dt[row['id']][missing.path] = node
                        stream.write(json.dumps({'id': row['id'], 'kind': 'solver', 'path': missing.path, 'response': node}) + '\n')
                        stream.flush()
                        acquired += 1
        for weight, penalty, continuous in product((0., .5, 1.), (0., .01, .02, .04, .06, .08, .10, .16, .24), (False, True)):
            settings = {'context_weight': weight, 'penalty': penalty, 'science_weight': 1.,
                        'reuse_review': True, 'late_feedback': True, 'plan_reviews': False,
                        'continuous_filter': continuous, 'threshold': 0.}
            # The shared encoder runs serially on the local device. Only missing
            # solver requests need remote concurrency, so this avoids MPS races.
            items = [one(row, settings) for row in dev]
            candidates.append({'settings': settings, 'metrics': statistics(items)})
            if len(candidates) % 9 == 0:
                best = max(candidates, key=lambda c: (c['metrics']['accuracy'], -c['metrics']['mean_compute']))
                print('Arithmetic POMDP development candidates', len(candidates), 'best', json.dumps(best), flush=True)
        best = max(candidates, key=lambda c: (c['metrics']['accuracy'], -c['metrics']['mean_compute']))
    summary = {'n': len(dev), 'fresh_data_used': False, 'baselines': baseline, 'retained_B': b,
               'best_A': best, 'new_solver_nodes': acquired,
               'A_target': max(v['accuracy'] for v in baseline.values()) + .1}
    (args.output / 'search.json').write_text(json.dumps(candidates, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'examples', 'verifier', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    run(p.parse_args())
