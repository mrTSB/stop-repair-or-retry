"""Paired training-only medium reviews after one continuation, with matched solver caps."""

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
from semantic_data import PRIMARY, read, write
from late_pairs import merged_records
from difficulty_pomdp import FIXED
from long_review_pilot import review
from reasoning_pilot import reasoning_prompt
from review_pilot import inference
import extended_acquire as old
from scoring import correct
from transport import resilient

SEED = 2026112053
APP = 'srr-followup-solver-20261120'
PROTOCOL = 'medium-late-compressed-v1'


def fingerprint(raw):
    return hashlib.sha256(raw).hexdigest()


def feedback(node):
    # This packet contains observed fields only. The complete paid scratchpad
    # remains available to the controller, while the compact solver receives
    # the canonical proposal and at most60words of the decisive check.
    check = ' '.join(str(node['verdict'].get('check', '')).split()[:60])
    return 'Independent answer: ' + node['answer'] + '\nCheck: ' + check


def selection(rows, trees, n):
    eligible = {r['id'] for split in ('train', 'dev')
                for r in read(PRIMARY / 'data/splits' / f'{split}-questions.jsonl')}
    candidates = []
    for row in rows:
        if row['id'] not in eligible:
            continue
        tree = trees[row['id']]
        path = min(('repair', 'retry', 'guided'),
                   key=lambda p: hashlib.sha256(f'{SEED}:parent:{row["id"]}:{p}'.encode()).digest())
        hard = int(sum(correct(tree[p], row['answer']) for p in FIXED) / len(FIXED) < .6)
        state = 2 * hard + int(correct(tree[path], row['answer']))
        candidates.append({'id': row['id'], 'path': path, 'subject': row['subject'], 'state': state})
    chosen = []
    used = set()
    for subject, count in (('math', n // 2), ('chemistry', n // 4), ('physics', n // 4)):
        group = sorted((r for r in candidates if r['subject'] == subject),
                       key=lambda r: hashlib.sha256(f'{SEED}:sample:{r["id"]}'.encode()).digest())
        selected = [r for state in range(4) for r in [v for v in group if v['state'] == state][:count // 4]]
        used.update(r['id'] for r in selected)
        selected += [r for r in group if r['id'] not in used][:count - len(selected)]
        if len(selected) != count:
            raise ValueError('Insufficient eligible training controls')
        chosen += selected
    assert len({r['id'] for r in chosen}) == len(chosen) == n
    return sorted(chosen, key=lambda r: hashlib.sha256(f'{SEED}:shuffle:{r["id"]}'.encode()).digest())


def scoped(row, path):
    return {**row, 'id': row['id'] + ':' + path + ':medium-late'}


def validate_review(row, path, current, node):
    calls = iter(node['calls'])
    consumed = []

    def replay(endpoint, payload):
        call = next(calls)
        assert endpoint == '/api/generate'
        assert fingerprint(payload['prompt'].encode()) == call['prompt_sha256']
        assert (payload['options']['seed'], payload['options']['num_predict'], payload['options']['num_ctx']) == (
            call['seed'], call['output_cap'], call['context_length'])
        consumed.append(call)
        return {'model': old.JUDGE, 'done': True, 'response': call['text'],
                'thinking': call.get('thinking', ''), 'prompt_eval_count': call['input_tokens'],
                'eval_count': call['output_tokens'], 'done_reason': call['finish'],
                'srr_backend': call.get('backend')}

    regenerated = review(replay, scoped(row, path), current, 1024, 512)
    assert len(consumed) == len(node['calls']) == node['physical_api_calls']
    for key in ('text', 'answer', 'raw_verdict', 'verdict', 'parse_valid', 'json_complete',
                'physical_api_calls', 'input_tokens', 'output_tokens', 'output_cap',
                'p_correct', 'action', 'prompt_sha256', 'model', 'model_digest'):
        assert regenerated[key] == node[key], key
    assert node['protocol'] == PROTOCOL


def collect(args):
    import modal
    rows, trees, judges = merged_records(args.data, 'train', False)
    by_id = {r['id']: r for r in rows}
    chosen = selection(rows, trees, args.n)
    args.output.mkdir(parents=True, exist_ok=True)
    contract = {'seed': SEED, 'split': 'training only', 'fresh_data_used': False,
                'selection': chosen, 'current_state_balance': dict(Counter(str(r['state']) for r in chosen)),
                'parent_paths': dict(Counter(r['path'] for r in chosen)),
                'selection_rule': 'Old fully paired training questions only. One hash-chosen parent per question; stratify training states, never test states.',
                'think_cap': 1024, 'answer_cap': 512, 'solver_cap': 256,
                'solver_template': 'Original repair_prompt; observed proposal plus truncated check packet.',
                'solver_digest': old.MODEL_DIGEST, 'reviewer_digest': old.JUDGE_DIGEST,
                'source_sha256': fingerprint(Path(__file__).read_bytes()),
                'review_source_sha256': fingerprint(Path(review.__code__.co_filename).read_bytes())}
    path = args.output / 'contract.json'
    if path.exists() and json.loads(path.read_bytes()) != contract:
        raise ValueError('Changed training collection contract')
    path.write_text(json.dumps(contract, indent=2) + '\n')
    request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
    tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    assert tags[old.MODEL] == old.MODEL_DIGEST and tags[old.JUDGE] == old.JUDGE_DIGEST
    with (args.output / 'train.jsonl').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        saved = [json.loads(s) for s in stream]
        known = {(r['id'], r['kind'], r['path']): r['response'] for r in saved}
        assert len(known) == len(saved)
        lock = threading.Lock()
        completed = 0

        def store(row, kind, path, node):
            with lock:
                stream.write(json.dumps({'id': row['id'], 'kind': kind, 'path': path, 'response': node}) + '\n')
                stream.flush()
                known[row['id'], kind, path] = node
            return node

        def one(item):
            nonlocal completed
            qid, path = item['id'], item['path']
            full = by_id[qid]
            row = {k: full[k] for k in ('id', 'domain', 'subject', 'question')}
            row['answer'] = ''
            current = trees[qid][path]
            key = qid, 'judge', path
            node = known.get(key)
            if node is None:
                node = review(request, scoped(row, path), current, 1024, 512)
                node['protocol'] = PROTOCOL
                node['backend'] = node['calls'][0]['backend']
                assert all(c['backend']['app'] == APP for c in node['calls'])
                node = store(row, 'judge', path, node)
            validate_review(row, path, current, node)
            prompt = old.repair_prompt(row, current, feedback(node))
            leaf = path + '/reviewed'
            if (qid, 'solver', leaf) not in known:
                final = inference(request, row, prompt, 'brief', False, path + '/repair')
                final['protocol'] = PROTOCOL
                store(row, 'solver', leaf, final)
            final = known[qid, 'solver', leaf]
            assert final['prompt_sha256'] == fingerprint(prompt.encode())
            assert final['model_digest'] == old.MODEL_DIGEST and final['output_cap'] == 256
            with lock:
                completed += 1
                if completed % 8 == 0 or completed == len(chosen):
                    print(f'Paired training medium review {completed}/{len(chosen)}', flush=True)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one, chosen))
    summary = {'n': len(chosen), 'split': 'training only', 'fresh_data_used': False,
               'methods': {}, 'by_subject': {}}
    for subject in ('all', 'math', 'chemistry', 'physics'):
        subset = [r for r in chosen if subject == 'all' or r['subject'] == subject]
        methods = {}
        for name in ('standard', 'medium'):
            reviews = [judges[r['id'], r['path']] if name == 'standard' else known[r['id'], 'judge', r['path']] for r in subset]
            leaves = [trees[r['id']][r['path'] + '/reviewed'] if name == 'standard' else known[r['id'], 'solver', r['path'] + '/reviewed'] for r in subset]
            methods[name] = {'n': len(subset),
                'review_correct': sum(correct(v, by_id[r['id']]['answer']) for r, v in zip(subset, reviews)),
                'repair_correct': sum(correct(v, by_id[r['id']]['answer']) for r, v in zip(subset, leaves)),
                'mean_review_tokens': sum(v['input_tokens'] + v['output_tokens'] for v in reviews) / len(subset),
                'mean_solver_tokens': sum(v['input_tokens'] + v['output_tokens'] for v in leaves) / len(subset),
                'mean_physical_review_calls': sum(v.get('physical_api_calls', 1) for v in reviews) / len(subset)}
        if subject == 'all':
            summary['methods'] = methods
        else:
            summary['by_subject'][subject] = methods
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n', type=int, default=192)
    args = parser.parse_args()
    if args.n % 16:
        parser.error('Use a multiple of16 for subject and hidden-state quotas')
    collect(args)
