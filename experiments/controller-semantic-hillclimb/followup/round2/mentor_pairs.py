"""Collect retained-review counterfactuals on a registered mixed training cohort.

Both doses use the exact same frozen compact solver and LLM. Previously paid
root reviews are reused exactly. The second repair reuses its root review.
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
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import ordinary, committed
from semantic_data import read
from reasoning_pilot import short_repair
from long_review_pilot import review
from transfer_variants import actor
from transport import resilient
import extended_acquire as old

SEED = 2026112123
APP = 'srr-followup-solver-20261120'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(qid, path):
    logical = path.replace('guided', 'repair').replace('reuse', 'repair')
    return int(hashlib.sha256(f'{old.SEED}:{qid}:{logical}'.encode()).hexdigest()[:8], 16) % 2**31


def cohort(old_rows, new_rows, old_selection):
    by_id = {r['id']: r for r in old_rows}
    old_pool = [by_id[r['id']] for r in old_selection]
    selected = []
    for pool, counts, label in ((old_pool, (32, 16, 16), 'old'), (new_rows, (64, 32, 32), 'new')):
        for subject, count in zip(('math', 'chemistry', 'physics'), counts):
            values = [r for r in pool if r['subject'] == subject]
            if label == 'old':
                values = sorted(values, key=lambda r: hashlib.sha256(f'{SEED}:{r["id"]}'.encode()).digest())
            # The new rows already have registered random order. Prefix quotas
            # let paired work overlap collection without conditioning on outcomes.
            if len(values) < count:
                raise ValueError('Insufficient registered training questions')
            selected.extend({'id': r['id'], 'subject': subject, 'bank': label} for r in values[:count])
    if len(selected) != 192 or len({r['id'] for r in selected}) != 192:
        raise ValueError('Use exactly192disjoint training questions')
    return sorted(selected, key=lambda r: hashlib.sha256(f'{SEED}:order:{r["id"]}'.encode()).digest())


def verify_review(row, current, node, dose, answer_cap=512):
    calls = iter(node['calls'])
    used = []
    def replay(endpoint, payload):
        call = next(calls)
        assert endpoint == '/api/generate'
        assert hashlib.sha256(payload['prompt'].encode()).hexdigest() == call['prompt_sha256']
        assert (payload['options']['seed'], payload['options']['num_predict'], payload['options']['num_ctx']) == (
            call['seed'], call['output_cap'], call['context_length'])
        used.append(call)
        return {'done': True, 'model': old.JUDGE, 'response': call['text'], 'thinking': call.get('thinking', ''),
                'prompt_eval_count': call['input_tokens'], 'eval_count': call['output_tokens'],
                'done_reason': call['finish'], 'srr_backend': call.get('backend')}
    actual = review(replay, row, current, dose, answer_cap)
    assert len(used) == len(node['calls']) == node['physical_api_calls']
    for name in ('text', 'answer', 'verdict', 'raw_verdict', 'input_tokens', 'output_tokens',
                 'prompt_sha256', 'parse_valid', 'json_complete', 'p_correct', 'action'):
        if actual[name] != node[name]:
            raise ValueError('Changed bounded review: ' + name)


def run(args):
    import modal
    old_rows, old_trees, _, _ = ordinary(args.old_data, 'train')
    new_rows = read(args.new_data / 'splits/train-questions.jsonl')
    old_selection = json.loads((args.old_selection / 'contract.json').read_bytes())['selection']
    selected = cohort(old_rows, new_rows, old_selection)
    extra_controls = {(r['id'], r['path']): r['response']
                      for r in committed(args.old_selection / 'train.jsonl') if r['kind'] == 'solver'}
    all_rows = {r['id']: r for r in old_rows + new_rows}
    shared_retry = {}
    for item in selected:
        if item['bank'] != 'old':
            continue
        qid = item['id']
        node = old_trees[qid].get('repair/retry', extra_controls.get((qid, 'guided/retry')))
        if node is None:
            raise ValueError('Missing a shared pure-retry training control')
        expected = hashlib.sha256(old.retry_prompt({**all_rows[qid], 'answer': ''}).encode()).hexdigest()
        if (node['model_digest'], node['prompt_sha256'], node['seed']) != (
                old.MODEL_DIGEST, expected, seed_for(qid, 'guided/retry')):
            raise ValueError('Incompatible shared pure-retry control')
        shared_retry[qid] = node
    contract = {'source_sha256': sha(__file__), 'seed': SEED, 'selection': selected,
                'old_training_sha256': sha(args.old_data / 'train-observations.jsonl'),
                'old_root_control_bank_sha256': sha(args.old_selection / 'train.jsonl'),
                'new_question_split_sha256': sha(args.new_data / 'splits/train-questions.jsonl'),
                'old_reuse_bank_sha256': sha(args.old_reuse / 'train.jsonl'),
                'questions': 192, 'subjects': dict(Counter(r['subject'] for r in selected)),
                'think_caps': [1024], 'answer_cap': 512, 'guided_cap': 128,
                'solver_digest': old.MODEL_DIGEST, 'reviewer_digest': old.JUDGE_DIGEST,
                'selection_rule': '64fixed-hash old pilot questions plus128registered-prefix new training questions, subject quotas fixed before inference.',
                'fresh_data_used': False, 'max_cloud_gpus': 2,
                'counterfactuals': ['guided', 'guided/reuse', 'guided/retry', 'retry/reuse'],
                'reuse': 'Same root review reused after either guided or plain retry; exact root controls and pure retry responses reused without outcome filtering.'}
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = args.output / 'contract.json'
    if recipe.exists() and json.loads(recipe.read_bytes()) != contract:
        raise ValueError('Changed mixed mentor-pair contract')
    recipe.write_text(json.dumps(contract, indent=2) + '\n')
    controls = {}
    for r in committed(args.old_selection / 'train.jsonl'):
        if r['kind'] == 'judge' and not r['path']:
            controls[r['id'], 1024, 'judge', ''] = r['response']
        elif r['kind'] == 'solver' and r['path'] == 'guided':
            controls[r['id'], 1024, 'solver', 'guided'] = r['response']
    for r in committed(args.old_reuse / 'train.jsonl'):
        controls[r['id'], 1024, 'solver', r['path']] = r['response']
    request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
    actual = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    assert actual[old.MODEL] == old.MODEL_DIGEST and actual[old.JUDGE] == old.JUDGE_DIGEST
    path = args.output / 'train.jsonl'
    known = {(r['id'], r['dose'], r['kind'], r['path']): r['response'] for r in committed(path)}
    assert len(known) == len(committed(path))
    lock = threading.Lock()
    done = 0
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def add(qid, dose, kind, where, function, reuse=None):
            key = qid, dose, kind, where
            if key not in known:
                node = reuse if reuse is not None else function()
                with lock:
                    stream.write(json.dumps({'id': qid, 'dose': dose, 'kind': kind, 'path': where,
                                             'response': node, 'exact_control_reused': reuse is not None}) + '\n')
                    stream.flush()
                    known[key] = node
            return known[key]

        def one(item):
            nonlocal done
            qid = item['id']
            row = {**all_rows[qid], 'answer': ''}
            if item['bank'] == 'old':
                tree = old_trees[qid]
            else:
                while True:
                    current = {r['path']: r['response'] for r in committed(args.new_data / 'train-observations.jsonl')
                               if r['id'] == qid and r['kind'] == 'solver'}
                    if all(p in current for p in ('', 'retry', 'repair/retry', 'retry/retry')):
                        tree = current
                        break
                    time.sleep(20)
            for dose in (1024,):
                teacher = add(qid, dose, 'judge', '', lambda: review(request, row, tree[''], dose, 512),
                              controls.get((qid, dose, 'judge', '')))
                verify_review(row, tree[''], teacher, dose, 512)
                prompt = short_repair(row, tree[''], teacher)
                guided = add(qid, dose, 'solver', 'guided',
                             lambda: actor(request, row, prompt, seed_for(qid, 'guided'), 128),
                             controls.get((qid, dose, 'solver', 'guided')))
                assert guided['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
                for where, current in (('guided/reuse', guided), ('retry/reuse', tree['retry'])):
                    prompt = short_repair(row, current, teacher)
                    node = add(qid, dose, 'solver', where,
                               lambda prompt=prompt, where=where: actor(request, row, prompt, seed_for(qid, where), 128),
                               controls.get((qid, dose, 'solver', where)))
                    assert (node['prompt_sha256'], node['seed'], node['output_cap']) == (
                        hashlib.sha256(prompt.encode()).hexdigest(), seed_for(qid, where), 128)
                pure_retry = shared_retry[qid] if item['bank'] == 'old' else tree['repair/retry']
                node = add(qid, dose, 'solver', 'guided/retry', lambda: None, pure_retry)
                assert node['seed'] == seed_for(qid, 'guided/retry')
            with lock:
                done += 1
                if done % 12 == 0 or done == 192:
                    print(f'Mixed retained-review training pairs {done}/192', flush=True)

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(one, selected))
    if len(known) != 192 * 5:
        raise ValueError('Incomplete mentor counterfactual bank')
    receipt = {'questions': 192, 'records': len(known), 'contract_sha256': sha(recipe),
               'bank_sha256': sha(path), 'fresh_data_used': False}
    (args.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('old-data', 'new-data', 'old-selection', 'old-reuse', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
