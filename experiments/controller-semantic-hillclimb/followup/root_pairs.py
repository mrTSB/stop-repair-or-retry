"""Expand the entire48-question root pilot to192paired training questions.

Every chosen root gets one medium review, an answer-first guided response and
both ordinary final continuations. No completed answer is retried by its grade.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from semantic_data import PRIMARY, read
from difficulty_pomdp import FIXED
from long_review_pilot import review
from reasoning_pilot import short_repair, inference
from banks import ordinary, committed, canonical_review
from medium_late_pairs import APP
from scoring import correct
from transport import resilient
import extended_acquire as old

SEED = 2026112067
PROTOCOL = 'medium-root-short-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected(rows, trees, pilot):
    by_id = {r['id']: r for r in rows}
    reused = sorted({r['id'] for r in pilot})
    if len(reused) != 48 or not set(reused) <= set(by_id):
        raise ValueError('Reuse the entire training-only48pilot')
    eligible = {r['id'] for split in ('train', 'dev')
                for r in read(PRIMARY / 'data/splits' / (split + '-questions.jsonl'))}
    candidates = []
    for row in rows:
        if row['id'] not in eligible or row['id'] in reused:
            continue
        tree = trees[row['id']]
        hard = int(sum(correct(tree[p], row['answer']) for p in FIXED) / len(FIXED) < .6)
        state = 2 * hard + int(correct(tree[''], row['answer']))
        candidates.append({'id': row['id'], 'subject': row['subject'], 'state': state})
    chosen = [{'id': qid, 'subject': by_id[qid]['subject'], 'reused_pilot': True} for qid in reused]
    for subject, n in (('math', 72), ('chemistry', 36), ('physics', 36)):
        group = sorted((r for r in candidates if r['subject'] == subject),
                       key=lambda r: hashlib.sha256(f'{SEED}:{r["id"]}'.encode()).digest())
        values = [r for state in range(4) for r in [v for v in group if v['state'] == state][:n // 4]]
        used = {r['id'] for r in values}
        values += [r for r in group if r['id'] not in used][:n - len(values)]
        if len(values) != n:
            raise ValueError('Insufficient old paired training questions')
        chosen += [{**r, 'reused_pilot': False} for r in values]
    if len(chosen) != 192 or len({r['id'] for r in chosen}) != 192:
        raise ValueError('Unexpected selection')
    return sorted(chosen, key=lambda r: hashlib.sha256(f'{SEED}:order:{r["id"]}'.encode()).digest())


def validate_review(row, current, node):
    calls = iter(node['calls'])
    consumed = []

    def replay(endpoint, payload):
        call = next(calls)
        assert endpoint == '/api/generate'
        assert hashlib.sha256(payload['prompt'].encode()).hexdigest() == call['prompt_sha256']
        assert (payload['options']['seed'], payload['options']['num_predict'], payload['options']['num_ctx']) == (
            call['seed'], call['output_cap'], call['context_length'])
        consumed.append(call)
        return {'done': True, 'model': old.JUDGE, 'response': call['text'], 'thinking': call.get('thinking', ''),
                'prompt_eval_count': call['input_tokens'], 'eval_count': call['output_tokens'],
                'done_reason': call['finish'], 'srr_backend': call.get('backend')}

    actual = review(replay, row, current, 1024, 512)
    assert len(consumed) == len(node['calls']) == node['physical_api_calls']
    for key in ('text', 'answer', 'verdict', 'raw_verdict', 'parse_valid', 'json_complete',
                'physical_api_calls', 'input_tokens', 'output_tokens', 'prompt_sha256', 'p_correct', 'action'):
        assert actual[key] == node[key], key


def collect(args):
    import modal
    rows, trees, original, _ = ordinary(args.data, 'train')
    by_id = {r['id']: r for r in rows}
    pilot = committed(args.pilot)
    chosen = selected(rows, trees, pilot)
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = {'seed': SEED, 'selection': chosen, 'pilot_sha256': sha(args.pilot),
              'questions': 192, 'subjects': dict(Counter(r['subject'] for r in chosen)),
              'protocol': PROTOCOL, 'fresh_data_used': False, 'source_sha256': sha(__file__),
              'rule': 'Reuse every48pilot question, add144hash-selected training questions balanced by training state; collect every chosen counterfactual before evaluation.',
              'teacher': {'think_cap': 1024, 'answer_cap': 512, 'context': 8192},
              'guided_actor': {'format': 'Final first,35justification words', 'cap': 128},
              'plain_continuations': ['guided/repair', 'guided/retry'],
              'solver_digest': old.MODEL_DIGEST, 'reviewer_digest': old.JUDGE_DIGEST}
    contract = args.output / 'contract.json'
    if contract.exists() and json.loads(contract.read_bytes()) != recipe:
        raise ValueError('Changed root-pair collection contract')
    contract.write_text(json.dumps(recipe, indent=2) + '\n')
    reusable = {(r['id'], r['kind']): r['response'] for r in pilot}
    request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
    tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    assert tags[old.MODEL] == old.MODEL_DIGEST and tags[old.JUDGE] == old.JUDGE_DIGEST
    old.request = request
    with (args.output / 'train.jsonl').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = committed(args.output / 'train.jsonl')
        known = {(r['id'], r['kind'], r['path']): r['response'] for r in saved}
        assert len(known) == len(saved)
        lock = threading.Lock()
        completed = 0

        def store(qid, kind, path, node):
            node = {**node, 'protocol': PROTOCOL}
            if 'backend' not in node:
                node['backend'] = node['calls'][0]['backend']
            with lock:
                stream.write(json.dumps({'id': qid, 'kind': kind, 'path': path, 'response': node}) + '\n')
                stream.flush()
                known[qid, kind, path] = node
            return node

        def one(item):
            nonlocal completed
            qid = item['id']
            row = {**by_id[qid], 'answer': ''}
            tree = {p: n for p, n in trees[qid].items() if not p.startswith('guided')}
            node = known.get((qid, 'judge', ''))
            if node is None:
                cached = reusable.get((qid, 'bounded-review'))
                node = store(qid, 'judge', '', cached if cached is not None else review(request, row, tree[''], 1024, 512))
            validate_review(row, tree[''], node)
            prompt = short_repair(row, tree[''], node)
            final = known.get((qid, 'solver', 'guided'))
            if final is None:
                cached = reusable.get((qid, 'bounded-short-repair'))
                final = store(qid, 'solver', 'guided', cached if cached is not None
                              else inference(request, row, prompt, False, 128))
            assert final['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
            tree['guided'] = final
            for path in ('guided/repair', 'guided/retry'):
                if (qid, 'solver', path) not in known:
                    store(qid, 'solver', path, old.call(row, path, tree, {(qid, ''): node}, False))
                tree[path] = known[qid, 'solver', path]
                assert tree[path]['prompt_sha256'] == hashlib.sha256(old.solver_prompt(row, path, tree, {(qid, ''): node}).encode()).hexdigest()
            with lock:
                completed += 1
                if completed % 12 == 0 or completed == len(chosen):
                    print(f'Complete medium-root training pairs {completed}/192', flush=True)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one, chosen))
    completion = {'questions': len(chosen), 'responses': len(known),
                  'contract_sha256': sha(contract), 'bank_sha256': sha(args.output / 'train.jsonl'),
                  'fresh_data_used': False}
    (args.output / 'collection-complete.json').write_text(json.dumps(completion, indent=2) + '\n')
    missing_controls = sum('guided' not in trees[r['id']] for r in chosen)
    if missing_controls:
        print(json.dumps({**completion, 'historical_guided_controls_missing': missing_controls,
                          'next': 'Run root_controls.py to complete every matched historical control and summary.'}), flush=True)
        return
    methods = {}
    for name, getter in (
        ('original-guided', lambda r: trees[r['id']]['guided']),
        ('medium-guided', lambda r: known[r['id'], 'solver', 'guided']),
        ('medium-review', lambda r: known[r['id'], 'judge', ''])):
        values = [getter(r) for r in chosen]
        methods[name] = {'correct': sum(correct(n, by_id[r['id']]['answer']) for r, n in zip(chosen, values)),
                         'mean_tokens': sum(n['input_tokens'] + n['output_tokens'] for n in values) / len(values)}
    result = {'questions': 192, 'split': 'training only', 'fresh_data_used': False, 'methods': methods}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'pilot', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    collect(p.parse_args())
