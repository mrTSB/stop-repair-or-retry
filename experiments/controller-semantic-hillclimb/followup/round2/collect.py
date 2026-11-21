"""Collect all prespecified paired branches on new training/development data.

Never select or regenerate an answer by correctness. No test collection is
permitted here; prospective test collection needs its own frozen policy.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import ordinary, committed
from semantic_data import read
from late_pairs import judge_prompt, sanitized
from review_pilot import inference
from transport import resilient
import extended_acquire as old

APP = 'srr-followup-solver-20261120'
PLAIN = ('', 'repair', 'retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry')
GUIDED = ('guided', 'guided/repair', 'guided/retry')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    import modal
    selection = json.loads((args.data / 'splits/selection.json').read_bytes())
    rows = read(args.data / 'splits' / (args.split + '-questions.jsonl'))
    assert len(rows) == selection['splits'][args.split]['n']
    assert sha(args.data / 'splits' / (args.split + '-questions.jsonl')) == selection['splits'][args.split]['sha256']
    contract = {'source_sha256': sha(__file__), 'split_sha256': selection['splits'][args.split]['sha256'],
                'solver_digest': old.MODEL_DIGEST, 'reviewer_digest': old.JUDGE_DIGEST,
                'mode': args.mode, 'plain_paths': PLAIN, 'ordinary_guided_paths': GUIDED,
                'initial_cap': old.INITIAL_CAP, 'continuation_cap': old.SOLVER_CAP,
                'review_cap': old.JUDGE_CAP, 'app': APP, 'max_cloud_gpus': 2,
                'reference_answers_used_for_inference': False}
    contract = json.loads(json.dumps(contract))
    receipt = args.data / (args.split + '-' + args.mode + '-contract.json')
    if receipt.exists() and json.loads(receipt.read_bytes()) != contract:
        raise ValueError('Changed paired collection contract')
    receipt.write_text(json.dumps(contract, indent=2) + '\n')
    worker = modal.Cls.from_name(APP, 'Ollama')()
    worker.request.autoscaler.update(max_containers=2)
    request = resilient(worker, args.data / 'transport-errors.jsonl')
    actual = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    assert actual[old.MODEL] == old.MODEL_DIGEST and actual[old.JUDGE] == old.JUDGE_DIGEST
    old.request = request
    base_path = args.data / (args.split + '-observations.jsonl')
    late_path = args.data / (args.split + '-late-observations.jsonl')
    known = {(r['id'], r['kind'], r['path']): r['response'] for r in committed(base_path)}
    late_known = {(r['id'], r['kind'], r['path']): r['response'] for r in committed(late_path)}
    assert len(known) == len(committed(base_path)) and len(late_known) == len(committed(late_path))
    done = 0
    lock = threading.Lock()
    with ExitStack() as stack:
        base_stream = stack.enter_context(base_path.open('a'))
        late_stream = stack.enter_context(late_path.open('a'))
        for stream in (base_stream, late_stream):
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def add(qid, kind, path, function, late=False):
            bank, stream = (late_known, late_stream) if late else (known, base_stream)
            key = qid, kind, path
            if key not in bank:
                node = function()
                with lock:
                    stream.write(json.dumps({'id': qid, 'kind': kind, 'path': path, 'response': node}) + '\n')
                    stream.flush()
                    bank[key] = node
            return bank[key]

        def one(full):
            nonlocal done
            row = {k: v for k, v in full.items() if k != 'answer'}
            row['answer'] = ''
            qid = row['id']
            tree, judges = {}, {}
            for path in PLAIN:
                tree[path] = add(qid, 'solver', path, lambda path=path: old.call(row, path, tree, judges, False))
                assert tree[path]['prompt_sha256'] == hashlib.sha256(old.solver_prompt(row, path, tree, judges).encode()).hexdigest()
            if args.mode == 'ordinary':
                judges[qid, ''] = add(qid, 'judge', '', lambda: old.call(row, '', tree, judges, True))
                for path in GUIDED:
                    tree[path] = add(qid, 'solver', path, lambda path=path: old.call(row, path, tree, judges, False))
                for path in ('repair', 'retry', 'guided'):
                    def review(path=path):
                        node = sanitized(inference(request, row, judge_prompt(row, path, tree), 'brief', True, path))
                        node['protocol'] = 'late-reviewed-repair-v1'
                        return node
                    judge = add(qid, 'judge', path, review, True)
                    def repair(path=path, judge=judge):
                        node = inference(request, row, old.repair_prompt(row, tree[path], old.feedback(judge)),
                                         'brief', False, path + '/repair')
                        node['protocol'] = 'late-reviewed-repair-v1'
                        return node
                    add(qid, 'solver', path + '/reviewed', repair, True)
            with lock:
                done += 1
                if done % 16 == 0 or done == len(rows):
                    print(f'{args.split} {args.mode}: {done}/{len(rows)} paired questions; {len(known) + len(late_known)} receipts', flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(one, rows))
    validated, _, _, _ = ordinary(args.data, args.split)
    assert len(validated) == len(rows)
    result = {'questions': len(rows), 'contract_sha256': sha(receipt),
              'base_bank_sha256': sha(base_path), 'late_bank_sha256': sha(late_path),
              'base_records': len(known), 'late_records': len(late_known), 'references_supplied': False}
    (args.data / (args.split + '-' + args.mode + '-complete.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--split', choices=('train', 'dev'), required=True)
    p.add_argument('--mode', choices=('plain', 'ordinary'), default='plain')
    p.add_argument('--workers', type=int, choices=(1, 2, 4), default=4)
    run(p.parse_args())
