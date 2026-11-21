"""Independent LLM-only B with the hybrid's same retained-review protocol.

Every question buys the same1024/512root review. Rules use only that review's
action/confidence and agreement with the paid compact answer. No POMDP,
neural observer, future response, or reference answer chooses an action.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import inspect
from itertools import product
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import canonical_review
from context_pomdp import equal, outcome
from difficulty_pomdp import MissingObservation
from long_review_pilot import review
from reasoning_pilot import short_repair
from transfer_variants import actor
from transport import resilient
from study_common import statistics, BASELINES
from study import fixed
import extended_acquire as old
import mentor_pairs

APP = 'srr-followup-solver-20261120'
ROOT_STOPS = ('action', 'agreement', .8, .95, .995, 1.)
SECOND_STOPS = ('agreement', 'never')
SECOND_ACTIONS = ('reuse', 'retry', 'root_action')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_review_charge(judge):
    calls = judge.get('calls', [])
    if len(calls) != judge.get('physical_api_calls') or len(calls) not in (1, 2):
        raise ValueError('The matched strong review requires every executed physical phase receipt')
    for name in ('input_tokens', 'output_tokens'):
        if judge[name] != sum(call[name] for call in calls):
            raise ValueError('Strong review token charge omits a physical phase')


def policy(row, tree, judges, settings):
    qid = row['id']
    if (qid, '') not in judges:
        raise MissingObservation('judge', '')
    judge = judges[qid, '']
    validate_review_charge(judge)
    rule = settings['root_stop']
    action = judge['action']
    if rule == 'agreement':
        action = 'stop' if equal(judge['answer'], tree['']['answer']) else 'repair'
    elif rule != 'action':
        action = 'stop' if judge['p_correct'] >= rule else (action if action != 'stop' else 'repair')
    if action not in ('stop', 'repair', 'retry'):
        raise ValueError('Invalid LLM root action')
    visited, decisions = [''], [{'path': '', 'action': action, 'queried': True}]
    path = ''
    if action != 'stop':
        path = 'guided' if action == 'repair' else 'retry'
        if path not in tree:
            raise MissingObservation('solver', path)
        visited.append(path)
        stop = settings['second_stop'] == 'agreement' and equal(judge['answer'], tree[path]['answer'])
        if stop:
            decisions.append({'path': path, 'action': 'stop', 'queried': False})
        else:
            second = settings['second_action']
            if second == 'root_action':
                second = 'retry' if judge['action'] == 'retry' else 'reuse'
            if second not in ('reuse', 'retry'):
                raise ValueError('Unknown retained-review continuation')
            decisions.append({'path': path, 'action': 'repair' if second == 'reuse' else 'retry', 'queried': False})
            path = path + '/' + second
            if path not in tree:
                raise MissingObservation('solver', path)
            visited.append(path)
    item = outcome(row, tree, judges, path, visited, [''], decisions)
    item.update(judge_physical_calls=judge['physical_api_calls'], verifier_calls=0,
                verifier_tokens=0, verifier_token_slots=0, verifier_parameters_billions=0.)
    if item['solver_calls'] > 3 or item['judge_calls'] != 1:
        raise ValueError('Matched B violated its retained-review horizon')
    return item


def metrics(items):
    result = statistics(items)
    result['mean_judge_physical_calls'] = sum(i['judge_physical_calls'] for i in items) / len(items)
    return result


def choose(ordinary, matched):
    # Keep the genuine accuracy winner; lower compute breaks accuracy ties.
    return max((ordinary, matched), key=lambda c: (c['dev']['accuracy'], -c['dev']['mean_compute']))


def copy_completed_bank(args, identity):
    source_summary = args.hybrid / 'summary.json'
    while not source_summary.exists():
        time.sleep(30)
    summary = json.loads(source_summary.read_bytes())
    source_identity = json.loads((args.hybrid / 'identity.json').read_bytes())
    if (summary['development_questions'] != 256 or source_identity['dev_questions_sha256'] != identity['dev_questions_sha256']):
        raise ValueError('Completed hybrid uses a different development panel')
    source = args.hybrid / 'dev-observations.jsonl'
    with source.open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        payload = stream.read()
    if (payload and not payload.endswith(b'\n')) or hashlib.sha256(payload).hexdigest() != summary['bank_sha256']:
        raise ValueError('Hybrid bank is incomplete or changed after its summary')
    receipt = {'source_summary_sha256': sha(source_summary), 'source_bank_sha256': summary['bank_sha256'],
               'copied_bytes': len(payload), 'source_bank': str(source.resolve()),
               'reuse': 'All exact completed hybrid responses; no outcome filtering; original bank remains read-only.'}
    destination = args.output / 'dev-observations.jsonl'
    provenance = args.output / 'copy.json'
    if provenance.exists():
        if json.loads(provenance.read_bytes()) != receipt or not destination.read_bytes().startswith(payload):
            raise ValueError('Changed matched B response provenance')
    else:
        if destination.exists():
            raise ValueError('Existing destination lacks its copy receipt')
        destination.write_bytes(payload)
        provenance.write_text(json.dumps(receipt, indent=2) + '\n')
    return destination


def run(args):
    # Reuse the hybrid's strict receipt validator, but never instantiate or
    # evaluate its POMDP/observer. All actions below call only policy().
    from development_mixed_mentor import development_view
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {'source_sha256': sha(__file__),
                'receipt_validator_sha256': sha(inspect.getfile(development_view)),
                'dev_questions_sha256': sha(args.data / 'splits/dev-questions.jsonl'),
                'hybrid_source': str(args.hybrid.resolve()), 'ordinary_B_source': str(args.ordinary.resolve()),
                'root_stop_rules': list(ROOT_STOPS), 'second_stop_rules': list(SECOND_STOPS),
                'second_actions': list(SECOND_ACTIONS), 'review_caps': [1024, 512], 'guided_cap': 128,
                'maximum_reviews': 1, 'maximum_additional_solver_calls': 2,
                'controller': 'LLM action/confidence/agreement rules only; no POMDP or neural features',
                'selection': 'All256development questions; highest accuracy then lowest compute; ordinary B retained',
                'test_used': False, 'maximum_cloud_gpus': 2}
    path = args.output / 'identity.json'
    if path.exists() and json.loads(path.read_bytes()) != identity:
        raise ValueError('Changed matched-review B protocol')
    path.write_text(json.dumps(identity, indent=2) + '\n')
    bank = copy_completed_bank(args, identity)
    raw = bank.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Partial matched B receipt requires explicit recovery')
    rows, trees, judges = development_view(args.data, bank)
    if len(rows) != 256:
        raise ValueError('Use the whole registered development panel')
    request = None
    if args.collect:
        import modal
        request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
        tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
        if tags.get(old.MODEL) != old.MODEL_DIGEST or tags.get(old.JUDGE) != old.JUDGE_DIGEST:
            raise ValueError('Changed compact solver or review weights')
    lock = threading.Lock()
    new_nodes = 0
    candidates = []
    with bank.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def acquire(row, missing):
            nonlocal new_nodes
            qid, kind, where = row['id'], missing.kind, missing.path
            safe, tree = {**row, 'answer': ''}, trees[qid]
            reused = False
            if kind == 'solver' and where == 'guided/retry':
                node = tree['repair/retry']
                if node['seed'] != mentor_pairs.seed_for(qid, where):
                    raise ValueError('Pure retry control has a different seed')
                reused = True
            elif request is None:
                raise RuntimeError('Missing matched B response; enable --collect') from missing
            elif kind == 'judge' and where == '':
                node = canonical_review(review(request, safe, tree[''], 1024, 512))
                mentor_pairs.verify_review(safe, tree[''], node, 1024, 512)
                validate_review_charge(node)
            elif kind == 'solver' and where in ('guided', 'guided/reuse', 'retry/reuse'):
                parent = '' if where == 'guided' else where.rsplit('/', 1)[0]
                if parent not in tree or (qid, '') not in judges:
                    raise ValueError('Matched B requested an unpaid prefix')
                prompt = short_repair(safe, tree[parent], judges[qid, ''])
                node = actor(request, safe, prompt, mentor_pairs.seed_for(qid, where), 128)
                if (node['prompt_sha256'], node['seed'], node['output_cap']) != (
                        hashlib.sha256(prompt.encode()).hexdigest(), mentor_pairs.seed_for(qid, where), 128):
                    raise ValueError('New matched B solver receipt differs from its call')
            else:
                raise ValueError('Unexpected matched B acquisition outside two-step tree')
            with lock:
                stream.write(json.dumps({'id': qid, 'kind': kind, 'path': where, 'response': node,
                                         'exact_control_reused': reused}) + '\n')
                stream.flush()
                if kind == 'judge':
                    judges[qid, ''] = node
                else:
                    tree[where] = node
                new_nodes += 1
                if new_nodes % 16 == 0:
                    print(f'Collected{new_nodes}additional matched B development responses', flush=True)

        def evaluate(settings):
            def one(row):
                while True:
                    try:
                        return policy(row, trees[row['id']], judges, settings)
                    except MissingObservation as missing:
                        acquire(row, missing)
            with ThreadPoolExecutor(max_workers=2) as pool:
                return list(pool.map(one, rows))

        for root, stop, action in product(ROOT_STOPS, SECOND_STOPS, SECOND_ACTIONS):
            settings = {'root_stop': root, 'second_stop': stop, 'second_action': action}
            items = evaluate(settings)
            candidates.append({'settings': settings, 'family': 'B', 'protocol': 'matched-retained-1024-512-128',
                               'dev': metrics(items)})
            (args.output / 'search.json').write_text(json.dumps(candidates, indent=2) + '\n')
    # Recheck every reused and newly collected prompt/seed/physical review.
    development_view(args.data, bank)
    matched = max(candidates, key=lambda c: (c['dev']['accuracy'], -c['dev']['mean_compute']))
    while not args.ordinary.exists():
        time.sleep(30)
    ordinary = json.loads(args.ordinary.read_bytes())
    old_identity = json.loads((args.ordinary.parent / 'identity.json').read_bytes())
    if ordinary['development_questions'] != 256 or old_identity['dev_questions_sha256'] != identity['dev_questions_sha256']:
        raise ValueError('Ordinary B uses a different development panel')
    retained = {**ordinary['B'], 'protocol': 'ordinary-late', 'source_summary_sha256': sha(args.ordinary)}
    best = choose(retained, matched)
    baselines = {name: statistics([fixed(r, trees[r['id']], *settings) for r in rows])
                 for name, settings in BASELINES.items()}
    result = {'B': best, 'matched_B': matched, 'ordinary_B': retained, 'baselines': baselines,
              'development_questions': len(rows), 'test_used': False, 'new_nodes': new_nodes,
              'B_above_highest_baseline': best['dev']['accuracy'] - max(b['accuracy'] for b in baselines.values()),
              'bank_sha256': sha(bank), 'copy_receipt_sha256': sha(args.output / 'copy.json'),
              'comparison': 'Preserve highest independent B accuracy for every final A/B/C gap; no C results altered.'}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'hybrid', 'ordinary', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--collect', action='store_true')
    run(p.parse_args())
