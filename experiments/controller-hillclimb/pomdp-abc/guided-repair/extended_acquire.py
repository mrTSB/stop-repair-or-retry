"""Shared compact-solver trees with optional paid reviews, isolated from prior runs."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from acquire import (DATA, LAST, MODEL, MODEL_DIGEST, PREVIOUS,
                     ROOT, SCIENCE_SUFFIX, chat, confidence, digest, existing_questions,
                     gsm8k, paired_archive, parse_answer, read, request, science, stem)
from guided_solver import repair_prompt
from compact import prompt as compact_prompt

SEED = 2026111908
JUDGE = 'qwen3.5:35b'
JUDGE_DIGEST = '3460ffeede5453ead027dbd2f821b12ad0aa3de54630971993babdb2165221f7'
JUDGE_ACTIVE = 3.
JUDGE_TOTAL = 35.
INITIAL_CAP = 128
SOLVER_CAP = 256
JUDGE_CAP = 512
PATHS = ('', 'repair', 'retry', 'guided', 'repair/repair', 'repair/retry',
         'retry/repair', 'retry/retry', 'guided/repair', 'guided/retry')
CHECKPOINTS = PATHS[:4]


def select(rows, counts, label):
    chosen = []
    for subject, count in counts.items():
        candidates = sorted((r for r in rows if r['subject'] == subject),
                            key=lambda r: hashlib.sha256(f'{SEED}:{label}:{r["id"]}'.encode()).digest())
        if len(candidates) < count:
            raise ValueError(f'Insufficient {subject}: {len(candidates)} < {count}')
        chosen += candidates[:count]
    random.Random(f'{SEED}:{label}').shuffle(chosen)
    return chosen


def prepare(args):
    previous = ROOT / 'runs/controller-hillclimb/pomdp-abc'
    old_train = read(previous / 'splits/train-questions.jsonl')
    retired = read(previous / 'splits/test-questions.jsonl')
    pilots = {r['id'] for r in read(previous / 'judge-deliberate-pilot.jsonl')}
    train = [r for r in old_train if r['id'] in pilots]
    if len(train) != 48:
        raise ValueError('Training pilot membership changed')
    extra = select(retired, {'math': 24, 'chemistry': 12, 'physics': 12}, 'train')
    train += extra
    used_train = {r['id'] for r in extra}
    dev = select([r for r in retired if r['id'] not in used_train],
                 {'math': 60, 'chemistry': 30, 'physics': 30}, 'dev')
    used = {r['id'] for r in train + dev}
    train += select([r for r in retired if r['id'] not in used],
                    {'math': 48, 'chemistry': 24, 'physics': 24}, 'train-extra')

    history = existing_questions() + paired_archive()
    for base in (LAST, PREVIOUS, previous):
        for split in ('train', 'dev', 'test'):
            path = base / 'splits' / f'{split}-questions.jsonl'
            if path.exists():
                history += read(path)
    ids, texts = {r['id'] for r in history}, {stem(r['question']) for r in history}
    exclusions = json.loads((DATA.parent / 'provenance/selection-manifest.json').read_bytes())
    ids.update(f'gsm8k-train-{i:05d}' for i in exclusions['historical_math_training_exclusions'])
    ids.update(exclusions['historical_math_test_exclusions'])
    ids.update(exclusions['historical_science_exclusions'])
    ids.update(r['id'] for r in exclusions['source_quality_exclusions'])
    pool = gsm8k(args.math_train, 'train') + [r for rows in science(args.science).values() for r in rows]
    available = []
    for row in sorted(pool, key=lambda r: hashlib.sha256(f'{SEED}:pool:{r["id"]}'.encode()).digest()):
        text = stem(row['question'])
        if row['id'] not in ids and text not in texts:
            available.append(row)
            ids.add(row['id'])
            texts.add(text)
    test = select(available, {'math': 300, 'chemistry': 150, 'physics': 150}, 'test')
    args.data.mkdir(parents=True, exist_ok=False)
    (args.data / 'splits').mkdir()
    manifest = {'seed': SEED, 'sources': {'math_train': digest(args.math_train), 'science': digest(args.science)},
                'train': '48 earlier training pilots and 144 retired study-test questions.',
                'dev': '120 disjoint retired study-test questions; no longer a test panel.',
                'test': '600 unused question IDs and normalized stems. GSM8K training-source questions and MMLU-Pro chemistry/physics. Biology pool exhausted.',
                'splits': {}}
    for split, rows in [('train', train), ('dev', dev), ('test', test)]:
        path = args.data / 'splits' / f'{split}-questions.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        manifest['splits'][split] = {'n': len(rows), 'subjects': dict(Counter(r['subject'] for r in rows)), 'sha256': digest(path)}
    (args.data / 'splits/selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


def retry_prompt(row):
    unit = 'one option letter' if row['domain'] == 'science' else 'a number'
    return chat(row['question'].removesuffix(SCIENCE_SUFFIX) + '\n\n'
                'Work through the problem carefully. Check the assumptions, units, and calculations. '
                f'Give at most 90 words of scratch work, then a line "Final: X" where X is {unit}.')


def feedback(node):
    obj = node['verdict']
    return str(obj.get('solution', '')) + '\nIndependent answer: ' + str(obj.get('answer', '')) + '\n' + str(obj.get('check', ''))


def solver_prompt(row, path, tree, judges):
    if not path:
        return compact_prompt(row['domain'], row['question'])
    if path.endswith('retry'):
        return retry_prompt(row)
    parent = path.rsplit('/', 1)[0] if '/' in path else ''
    return repair_prompt(row, tree[parent], feedback(judges[row['id'], '']) if path == 'guided' else '')


def judge_prompt(row, path, tree):
    unit = 'one option letter' if row['domain'] == 'science' else 'a number'
    remaining = 2 - int(bool(path))
    repair_description = ('REPAIR asks the compact solver to revise it using your review' if not path
                          else 'REPAIR asks the compact solver to check its solution again')
    return chat(
        'Control a compact solver by independently checking the original problem. '
        'First solve it yourself, explicitly checking each calculation or scientific principle. '
        'Then compare the CURRENT solution with your independent solution. '
        'Evaluate whether the CURRENT final answer is correct; different reasoning can still give a correct answer. '
        f'STOP submits the CURRENT answer, {repair_description}, and RETRY starts a fresh compact solution. '
        f'You have {remaining} further solver calls. '
        'Choose the action that maximizes final answer accuracy. '
        'Return ONLY JSON with fields in this order: '
        '"solution" (your independent solution, at most 90 words), '
        f'"answer" (your final answer, {unit}), '
        '"check" (at most 24 words identifying the error or evidence the current solution is correct), '
        '"p_correct" (probability the CURRENT final answer is correct, from 0 to 1), '
        '"action" ("stop", "repair", or "retry").\n\n'
        + row['question'].removesuffix(SCIENCE_SUFFIX) + '\n\nCURRENT solution:\n' + tree[path]['text'])


def parse_verdict(text):
    decoder = json.JSONDecoder(strict=False)
    escaped = re.sub(r'\\(?!["\\/bfnrtu])', lambda _: '\\\\', text)
    for candidate_text in dict.fromkeys((text, escaped)):
        for match in re.finditer(r'\{', candidate_text):
            try:
                obj, _ = decoder.raw_decode(candidate_text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                p, action = obj.get('p_correct'), obj.get('action')
                valid = type(p) in (int, float) and 0 <= p <= 1 and action in ('stop', 'repair', 'retry')
                if valid:
                    return obj, True
    return {'solution': '', 'answer': '', 'check': '', 'p_correct': .5, 'action': 'repair'}, False


def call(row, path, tree, judges, is_judge=False):
    model, fingerprint = (JUDGE, JUDGE_DIGEST) if is_judge else (MODEL, MODEL_DIGEST)
    prompt = judge_prompt(row, path, tree) if is_judge else solver_prompt(row, path, tree, judges)
    logical = path.replace('guided', 'repair')
    seed = int(hashlib.sha256(f'{SEED}:{row["id"]}:{logical}'.encode()).hexdigest()[:8], 16) % 2**31
    started = time.monotonic()
    payload = {'model': model, 'prompt': prompt, 'raw': True,
               'stream': False, 'logprobs': not is_judge, 'keep_alive': '30m',
               'options': {'temperature': 0 if is_judge or not path else .65, 'seed': seed,
                           'num_predict': JUDGE_CAP if is_judge else (SOLVER_CAP if path else INITIAL_CAP), 'num_ctx': 4096}}
    response = request('/api/generate', payload)
    if not response.get('done') or response.get('model') != model:
        detail = {k: response.get(k) for k in ('model', 'done', 'done_reason', 'error', 'prompt_eval_count', 'eval_count')}
        raise ValueError('Incomplete inference result: '+json.dumps({'id': row['id'], 'path': path, 'expected_model': model, **detail}))
    text = response['response']
    node = {'model': model, 'model_digest': fingerprint, 'text': text,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'seed': seed, 'temperature': 0 if is_judge or not path else .65,
            'finish': response['done_reason'], 'input_tokens': int(response['prompt_eval_count']),
            'output_tokens': int(response.get('eval_count', 0)), 'latency_seconds': time.monotonic() - started}
    node['backend'] = response.get('srr_backend', {'provider': 'local_ollama', 'ollama_version': '0.19.0'})
    if is_judge:
        obj, valid = parse_verdict(text)
        node.update(verdict=obj, parse_valid=valid, p_correct=obj['p_correct'], action=obj['action'], response_format='text-json')
        node['answer'] = parse_answer(row['domain'], 'Final: ' + str(obj.get('answer', '')))
    else:
        if text and not response.get('logprobs'):
            raise ValueError('Solver token probabilities are missing')
        node.update(answer=parse_answer(row['domain'], text), confidence=confidence(text, response.get('logprobs', []), row['domain']))
    return node


def collect(args):
    global request
    if args.backend == 'modal':
        import modal
        worker = modal.Cls.from_name('srr-pomdp-abc-20261119', 'Ollama')()
        request = worker.request.remote
    rows = read(args.data / 'splits' / f'{args.split}-questions.jsonl')
    cache_path = args.data / 'initial-cache.jsonl'
    initial_cache = {r['id']: r['response'] for r in read(cache_path)} if cache_path.exists() else {}
    tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    if tags.get(MODEL) != MODEL_DIGEST or tags.get(JUDGE) != JUDGE_DIGEST:
        raise ValueError('Model weights changed')
    target = args.data / f'{args.split}-observations.jsonl'
    with target.open('a+', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        prior = [json.loads(line) for line in stream]
        known = {(r['id'], r['kind'], r['path']): r['response'] for r in prior}
        if len(known) != len(prior):
            raise ValueError('Duplicate observations')
        prior_by_id = {r['id']: {} for r in rows}
        for (i, kind, path), node in known.items():
            if i not in prior_by_id:
                raise ValueError('Unexpected collection question')
            prior_by_id[i][kind, path] = node
        lock, stopped = threading.Lock(), threading.Event()
        completed = 0

        def collect_one(row):
            nonlocal completed
            if stopped.is_set():
                return
            local = prior_by_id[row['id']]
            tree = {p: n for (k, p), n in local.items() if k == 'solver'}
            judges = {(row['id'], p): n for (k, p), n in local.items() if k == 'judge'}
            operations = [('solver', p) for p in ('', 'repair', 'retry')]
            if args.phase == 'full':
                operations += [('judge', '')] + [('solver', p) for p in PATHS[3:]]
                operations += [('judge', p) for p in CHECKPOINTS[1:]]
            for kind, path in operations:
                if (kind, path) in local:
                    continue
                if stopped.is_set():
                    return
                try:
                    node = initial_cache[row['id']] if kind == 'solver' and not path and row['id'] in initial_cache else call(row, path, tree, judges, kind == 'judge')
                except Exception as error:
                    with lock:
                        with (args.data / 'collection-errors.jsonl').open('a') as failures:
                            failures.write(json.dumps({'split': args.split, 'id': row['id'], 'kind': kind, 'path': path,
                                                       'error_type': type(error).__name__, 'error': str(error)})+'\n')
                    stopped.set()
                    raise
                with lock:
                    stream.write(json.dumps({'id': row['id'], 'kind': kind, 'path': path, 'response': node}) + '\n')
                    stream.flush()
                local[kind, path] = node
                if kind == 'judge':
                    judges[row['id'], path] = node
                else:
                    tree[path] = node
            with lock:
                completed += 1
                if completed % 8 == 0 or completed == len(rows):
                    print(f'{args.split} {args.phase} {completed}/{len(rows)}', flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for _ in executor.map(collect_one, rows):
                pass


def seed_initial(args):
    from study import records
    source = ROOT / 'runs/controller-hillclimb/pomdp-abc'
    roots = {}
    for split in ('train', 'test'):
        _, trees, _ = records(source, split)
        roots.update({key: tree[''] for key, tree in trees.items()})
    rows = read(args.data / 'splits/train-questions.jsonl') + read(args.data / 'splits/dev-questions.jsonl')
    target = args.data / 'initial-cache.jsonl'
    if target.exists():
        previous = {r['id']: r['response'] for r in read(target)}
        if any(roots[key] != node for key, node in previous.items()):
            raise ValueError('An archived initial response changed')
    with target.open('w') as stream:
        for row in rows:
            node = roots[row['id']]
            if node['prompt_sha256'] != hashlib.sha256(solver_prompt(row, '', {}, {}).encode()).hexdigest():
                raise ValueError('Archived initial prompt changed')
            stream.write(json.dumps({'id': row['id'], 'response': node}) + '\n')
    print(f'Copied {len(rows)} unchanged compact initial responses.')


def collect_holdout(args):
    global request
    import numpy as np
    from context_pomdp import evaluate
    from extended_study import evidence, methods
    from study import fixed

    frozen = json.loads(args.frozen.read_bytes())
    if frozen['evidence'] != evidence(args.data):
        raise ValueError('Code or selection evidence changed after freeze')
    if args.backend == 'modal':
        import modal
        worker = modal.Cls.from_name('srr-pomdp-abc-20261119', 'Ollama')()
        request = worker.request.remote
    tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    if tags.get(MODEL) != MODEL_DIGEST or tags.get(JUDGE) != JUDGE_DIGEST:
        raise ValueError('Model weights changed')
    policies = methods(frozen)
    questions = read(args.data / 'splits/test-questions.jsonl')
    target = args.data / 'test-observations.jsonl'
    metadata = args.data / 'test-collection.json'
    identity = {'freeze_sha256': digest(args.frozen), 'questions_sha256': digest(args.data / 'splits/test-questions.jsonl')}
    if metadata.exists() and any(json.loads(metadata.read_bytes()).get(k) != v for k, v in identity.items()):
        raise ValueError('Existing holdout belongs to a different frozen experiment')
    for _, model, settings, method in policies:
        if method != 'C':
            continue
        for domain, dynamics in model.dynamics.items():
            penalty = settings['penalty']*(settings.get('science_weight', 1.) if domain == 'science' else 1.)
            planned = dynamics.bellman(penalty)
            for remaining in (1, 2):
                for condition in range(6):
                    dynamics.information_value(.5, remaining, np.zeros(3), planned, penalty, condition, settings.get('reuse_review', False))
    with target.open('a+', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        prior = [json.loads(line) for line in stream]
        known = {(r['id'], r['kind'], r['path']): r['response'] for r in prior}
        if len(known) != len(prior):
            raise ValueError('Duplicate held-out observations')
        expected_ids = {r['id'] for r in questions}
        if any(i not in expected_ids for i, _, _ in known):
            raise ValueError('Unexpected holdout question')
        prior_by_id = {i: {} for i in expected_ids}
        for (i, kind, path), node in known.items():
            prior_by_id[i][kind, path] = node
        completed_ids = set(json.loads(metadata.read_bytes()).get('completed_ids', [])) if metadata.exists() else set()
        if not completed_ids <= expected_ids:
            raise ValueError('Unexpected completed holdout question')
        lock, stopped = threading.Lock(), threading.Event()

        def collect_one(question):
            if stopped.is_set() or question['id'] in completed_ids:
                return
            row = {**question, 'answer': ''}
            local = prior_by_id[row['id']]
            tree = {p: n for (k, p), n in local.items() if k == 'solver'}
            judges = {(row['id'], p): n for (k, p), n in local.items() if k == 'judge'}

            def ensure(kind, path):
                key = row['id'], kind, path
                if (kind, path) in local:
                    return
                if kind == 'solver' and path == 'guided':
                    ensure('judge', '')
                try:
                    node = call(row, path, tree, judges, kind == 'judge')
                except Exception:
                    stopped.set()
                    raise
                with lock:
                    stream.write(json.dumps({'id': row['id'], 'kind': kind, 'path': path, 'response': node}) + '\n')
                    stream.flush()
                    known[key] = node
                local[kind, path] = node
                if kind == 'solver':
                    tree[path] = node
                else:
                    judges[row['id'], path] = node

            ensure('solver', '')
            for name, config in frozen['baselines'].items():
                action = config['action']
                if action != 'stop':
                    for depth in range(1, config['depth'] + 1):
                        ensure('solver', '/'.join([action]*depth))
                fixed(row, tree, action, config['depth'])
            for _, model, settings, method in policies:
                for attempt in range(12):
                    try:
                        evaluate(row, tree, judges, model, settings, method)
                        break
                    except KeyError as missing:
                        key = missing.args[0]
                        if isinstance(key, tuple) and len(key) == 2 and key[0] == row['id'] and key[1] in CHECKPOINTS:
                            ensure('judge', key[1])
                        elif isinstance(key, str) and key in PATHS:
                            ensure('solver', key)
                        else:
                            raise
                else:
                    raise RuntimeError('Policy failed to converge during adaptive collection')
            with lock:
                completed_ids.add(row['id'])
                payload = {**identity, 'completed': len(completed_ids), 'completed_ids': sorted(completed_ids),
                           'questions': len(questions), 'observations': len(known),
                           'solver_observations': sum(k == 'solver' for _, k, _ in known),
                           'judge_observations': sum(k == 'judge' for _, k, _ in known)}
                temporary = metadata.with_suffix('.tmp')
                temporary.write_text(json.dumps(payload, indent=2) + '\n')
                temporary.replace(metadata)
                if len(completed_ids) % 10 == 0 or len(completed_ids) == len(questions):
                    print(f'Frozen fresh holdout {len(completed_ids)}/{len(questions)}; {len(known)} observations', flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for _ in executor.map(collect_one, questions):
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'seed', 'collect', 'holdout'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--math-train', type=Path)
    parser.add_argument('--science', type=Path)
    parser.add_argument('--split', choices=('train', 'dev', 'all'))
    parser.add_argument('--phase', choices=('initial', 'full'), default='initial')
    parser.add_argument('--frozen', type=Path)
    parser.add_argument('--backend', choices=('local', 'modal'), default='local')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('Use between one and eight collection workers')
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'seed':
        seed_initial(args)
    elif args.action == 'holdout':
        collect_holdout(args)
    elif args.split == 'all':
        for split in ('train', 'dev'):
            args.split = split
            collect(args)
    else:
        collect(args)


if __name__ == '__main__':
    main()
