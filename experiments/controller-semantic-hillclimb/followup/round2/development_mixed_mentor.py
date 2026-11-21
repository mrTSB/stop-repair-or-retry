"""Select the mixed-data repairability/retained-review hybrid on development only.

Strong root reviews and short repairs have their own private bank. All saved
physical review phases and solver passes remain charged by policy evaluation.
"""

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
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import ordinary, committed, canonical_review
from difficulty_pomdp import PLAIN, MissingObservation
from long_review_pilot import review
from reasoning_pilot import short_repair
from transfer_variants import actor
from study_common import statistics, BASELINES
from study import fixed
from transport import resilient
import extended_acquire as old
import mentor_pairs
import hybrid_repairability as semantic
from prepare_mixed_mentor import sha, plain_data

APP = 'srr-followup-solver-20261120'


def select(candidates, b_compute=None):
    """Keep the accuracy winner and independently enforce the actual B cost ceiling."""
    best = lambda values: max(values, key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute']))
    queried = [r for r in candidates if r['dev']['mean_judge_calls'] > 0]
    cheap = [r for r in queried if b_compute is not None and r['dev']['mean_compute'] < b_compute]
    return {'best': best(candidates), 'best_queried': best(queried) if queried else None,
            'selected_C': best(cheap) if cheap else None,
            'C_compute_ceiling': b_compute,
            'selection_rule': 'Highest development accuracy, then lowest compute; selected_C must buy reviews and use strictly less compute than the strongest independent B.'}


def development_view(data, bank):
    rows, original = plain_data(data, 'dev')
    trees = {r['id']: {p: n for p, n in original[r['id']].items() if p in PLAIN} for r in rows}
    by_id = {r['id']: r for r in rows}
    judges, seen = {}, set()
    records = committed(bank)
    for record in records:
        qid, kind, path, node = record['id'], record['kind'], record['path'], record['response']
        if qid not in by_id or (qid, kind, path) in seen:
            raise ValueError('Unexpected or duplicate mixed-mentor development response')
        seen.add((qid, kind, path))
        if kind == 'judge' and not path:
            judges[qid, path] = canonical_review(node)
        elif kind == 'solver' and path in ('guided', 'guided/reuse', 'retry/reuse', 'guided/retry'):
            trees[qid][path] = node
        else:
            raise ValueError('Unexpected mixed-mentor development path')
    for record in records:
        qid, kind, path, node = record['id'], record['kind'], record['path'], record['response']
        row, tree = {**by_id[qid], 'answer': ''}, trees[qid]
        if kind == 'judge':
            mentor_pairs.verify_review(row, tree[''], node, 1024, 512)
        else:
            if path == 'guided/retry':
                prompt, cap = old.retry_prompt(row), old.SOLVER_CAP
            else:
                parent = '' if path == 'guided' else path.rsplit('/', 1)[0]
                prompt, cap = short_repair(row, tree[parent], judges[qid, '']), 128
            actual_cap = node.get('output_cap', cap if path == 'guided/retry' else None)
            if (node['prompt_sha256'], node['seed'], node['model_digest'], actual_cap) != (
                    hashlib.sha256(prompt.encode()).hexdigest(), mentor_pairs.seed_for(qid, path), old.MODEL_DIGEST, cap):
                raise ValueError('Changed mixed-mentor development response')
            if not 0 <= node['output_tokens'] <= cap:
                raise ValueError('Mixed-mentor development solver exceeded its output contract')
    return rows, trees, judges


def run(args):
    while not (args.models / 'complete.json').exists():
        if not args.wait:
            raise ValueError('Finish the mixed-mentor fit or use --wait')
        time.sleep(30)
    fitted = json.loads((args.models / 'complete.json').read_bytes())
    models = fitted['models']
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {'source_sha256': sha(__file__), 'hybrid_sha256': sha(semantic.__file__),
                'models_sha256': sha(args.models / 'complete.json'),
                'dev_questions_sha256': sha(args.data / 'splits/dev-questions.jsonl'),
                'ordinary_plain_completion_sha256': sha(args.data / 'dev-plain-complete.json'),
                'source_population': 'Round2 GSM8K and SuperGPQA;256development questions only',
                'test_used': False, 'maximum_cloud_gpus': 2,
                'review': {'thinking_cap': 1024, 'answer_cap': 512, 'guided_cap': 128},
                'B_reference': str(args.b_reference.resolve()) if args.b_reference is not None else None,
                'penalties': [0., .02, .04, .08, .12, .18, .24], 'thresholds': [0., .03, .08]}
    receipt = args.output / 'identity.json'
    if receipt.exists() and json.loads(receipt.read_bytes()) != identity:
        raise ValueError('Changed development protocol; use a new output directory')
    receipt.write_text(json.dumps(identity, indent=2) + '\n')
    bank = args.output / 'dev-observations.jsonl'
    rows, trees, judges = development_view(args.data, bank)
    if len(rows) != 256:
        raise ValueError('Use all256registered development questions')
    request = None
    if args.collect:
        import modal
        request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
        tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
        if tags.get(old.MODEL) != old.MODEL_DIGEST or tags.get(old.JUDGE) != old.JUDGE_DIGEST:
            raise ValueError('Changed solver or reviewer')
    candidates, subject_candidates, new_nodes = [], {}, 0
    cache = {}
    cache_path = args.output / 'private-neural-cache.pkl'
    if cache_path.exists():
        cache = pickle.loads(cache_path.read_bytes())
    lock = threading.Lock()
    with bank.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def acquire(row, missing):
            nonlocal new_nodes
            qid, kind, path = row['id'], missing.kind, missing.path
            safe, tree = {**row, 'answer': ''}, trees[qid]
            reused = False
            if kind == 'solver' and path == 'guided/retry':
                node = tree['repair/retry']
                if node['seed'] != mentor_pairs.seed_for(qid, path):
                    raise ValueError('Pure retry reuse has a different seed')
                reused = True
            elif request is None:
                raise RuntimeError('Missing development response; enable --collect') from missing
            elif kind == 'judge' and path == '':
                node = canonical_review(review(request, safe, tree[''], 1024, 512))
            elif kind == 'solver' and path in ('guided', 'guided/reuse', 'retry/reuse'):
                parent = '' if path == 'guided' else path.rsplit('/', 1)[0]
                if parent not in tree or (qid, '') not in judges:
                    raise ValueError('The hybrid requested an unpaid prefix')
                node = actor(request, safe, short_repair(safe, tree[parent], judges[qid, '']),
                             mentor_pairs.seed_for(qid, path), 128)
            else:
                raise ValueError('Unexpected hybrid request outside the completed plain tree')
            with lock:
                stream.write(json.dumps({'id': qid, 'kind': kind, 'path': path, 'response': node,
                                         'exact_control_reused': reused}) + '\n')
                stream.flush()
                if kind == 'judge':
                    judges[qid, ''] = node
                else:
                    tree[path] = node
                new_nodes += 1
                if new_nodes % 16 == 0:
                    print(f'Collected {new_nodes} mixed-mentor development responses', flush=True)

        def evaluate(model, settings):
            def one(row):
                while True:
                    try:
                        return semantic.evaluate(row, trees[row['id']], judges, model, settings, 'C')
                    except MissingObservation as missing:
                        acquire(row, missing)
            with ThreadPoolExecutor(max_workers=2) as pool:
                return list(pool.map(one, rows))

        observer = None
        for spec in models:
            path = args.models / (spec['key'] + '.pkl')
            if sha(path) != spec['sha256']:
                raise ValueError('Changed fitted hybrid candidate')
            model = pickle.loads(path.read_bytes())
            model.base.classifier.device = args.device
            if observer is not None:
                for name in ('_model', '_tokenizer', '_recipe', '_cache', '_retriever', '_backgrounds'):
                    setattr(model.base.classifier, name, getattr(observer, name))
            else:
                model.base.classifier._cache = cache
            model.prepare()
            observer = model.base.classifier
            subject_candidates = {subject: [] for subject in ('math', 'chemistry', 'physics')}
            for penalty, threshold in product(identity['penalties'], identity['thresholds']):
                settings = {'penalty': penalty, 'threshold': threshold}
                items = evaluate(model, settings)
                candidate = {'model': spec['key'], 'settings': settings, 'dev': statistics(items)}
                candidates.append(candidate)
                for subject in subject_candidates:
                    subset = [item for row, item in zip(rows, items) if row['subject'] == subject]
                    subject_candidates[subject].append({'settings': settings, 'dev': statistics(subset)})
            for utility_penalty in (0., .02, .04, .08, .16):
                settings = {'by_subject': {subject: max(group, key=lambda r: (
                    r['dev']['accuracy'] - utility_penalty * r['dev']['mean_compute'] / 1000,
                    r['dev']['accuracy'], -r['dev']['mean_compute']))['settings']
                    for subject, group in subject_candidates.items()}}
                candidates.append({'model': spec['key'], 'settings': settings,
                                   'selection_utility_penalty': utility_penalty, 'dev': statistics(evaluate(model, settings))})
            (args.output / 'search.json').write_text(json.dumps(candidates, indent=2) + '\n')
            cache_path.write_bytes(pickle.dumps(observer._cache, protocol=5))
            best = max(candidates, key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute']))
            print('Finished hybrid', spec['key'], 'best', json.dumps(best), flush=True)
    baselines = {name: statistics([fixed(row, trees[row['id']], *parameters) for row in rows])
                 for name, parameters in BASELINES.items()}
    summary = {'development_questions': len(rows), 'baselines': baselines, **select(candidates),
               'new_development_nodes': new_nodes, 'test_used': False,
               'B_comparison': 'Compare against completed independent ordinary B on this same256development panel.',
               'bank_sha256': sha(bank), 'source_sha256': sha(__file__)}
    if args.b_reference is not None:
        # Collect and evaluate C first, preserving its complete search while
        # waiting for the independently evaluated ordinary B cost ceiling.
        (args.output / 'pending-B-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        if not args.b_reference.exists():
            print('Hybrid search complete; waiting for the independent B reference before selecting a cheaper C', flush=True)
        while not args.b_reference.exists():
            time.sleep(30)
        reference = json.loads(args.b_reference.read_bytes())
        reference_identity = json.loads((args.b_reference.parent / 'identity.json').read_bytes())
        if reference_identity['dev_questions_sha256'] != identity['dev_questions_sha256'] or reference['development_questions'] != len(rows):
            raise ValueError('B reference uses a different development panel')
        b = reference['B']
        summary.update(select(candidates, b['dev']['mean_compute']))
        summary.update(B=b, B_reference_sha256=sha(args.b_reference), A=fitted['base_A'],
                       B_comparison='Strongest independently selected ordinary LLM controller on the identical development panel.')
        chosen = summary['selected_C']
        if chosen is not None:
            higher = max(b['dev']['accuracy'], fitted['base_A']['dev']['accuracy'])
            summary['C_above_higher_A_B'] = chosen['dev']['accuracy'] - higher
            summary['C_plus_10_points'] = chosen['dev']['accuracy'] >= higher + .1 - 1e-12
            summary['C_compute_saved_vs_B'] = 1. - chosen['dev']['mean_compute'] / b['dev']['mean_compute']
        else:
            summary['C_plus_10_points'] = False
            summary['selection_status'] = 'No queried hybrid candidate is strictly cheaper than independent B.'
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'models', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    p.add_argument('--b-reference', type=Path, help='Wait for this independent B summary only after C acquisition/search finishes.')
    p.add_argument('--collect', action='store_true')
    p.add_argument('--wait', action='store_true')
    run(p.parse_args())
