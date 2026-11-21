"""Cache prespecified fixed-action responses on the entire912panel without grading.

This stage does not load controller models, compute correctness, print answers,
or select any questions. Reference answers are removed before every API call.
The frozen solver prompts/caps/seeds are independent of subsequent selection.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from semantic_data import read
import extended_acquire as old
from transport import resilient
from banks import committed
from medium_late_pairs import APP

PATHS = ('', 'repair', 'retry', 'repair/repair', 'retry/retry')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    import modal
    rows = read(args.data / 'splits/test-questions.jsonl')
    if len(rows) != 912 or len({r['id'] for r in rows}) != 912:
        raise ValueError('All912prespecified questions are required')
    ids = {r['id'] for r in rows}
    fitting = {r['id'] for split in ('train', 'dev') for r in read(args.data / 'splits' / (split + '-questions.jsonl'))}
    if ids & fitting:
        raise ValueError('Training/development overlap')
    recipe = {'test_sha256': sha(args.data / 'splits/test-questions.jsonl'),
              'selection_sha256': sha(args.data / 'splits/selection.json'),
              'questions': 912, 'paths': list(PATHS), 'solver_digest': old.MODEL_DIGEST,
              'source_sha256': sha(__file__), 'solver_source_sha256': sha(old.__file__),
              'reference_answers_sent': False, 'graded': False,
              'rule': 'Every reserved question; fixed prompts/caps/seeds; no controller or score is consulted.',
              'reused_same_source': {suffix: sha(args.previous / ('test-' + suffix + '.jsonl'))
                                     for suffix in ('observations', 'late-observations')}}
    contract = args.data / 'fixed-cache-contract.json'
    if contract.exists() and json.loads(contract.read_bytes()) != recipe:
        raise ValueError('Changed fixed-cache protocol')
    contract.write_text(json.dumps(recipe, indent=2) + '\n')
    # Preserve all paid old192responses, including ordinary reviews, byte-level
    # response fields unchanged. None of their grades have been inspected.
    for suffix in ('observations', 'late-observations'):
        path = args.data / ('test-' + suffix + '.jsonl')
        saved = committed(path)
        known = {(r['id'], r['kind'], r['path']): r['response'] for r in saved}
        if len(known) != len(saved):
            raise ValueError('Duplicate response key')
        with path.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for item in committed(args.previous / ('test-' + suffix + '.jsonl')):
                if item['id'] not in ids:
                    raise ValueError('Unexpected reused question')
                key = item['id'], item['kind'], item['path']
                if key in known:
                    if known[key] != item['response']:
                        raise ValueError('Changed reused response')
                else:
                    stream.write(json.dumps(item) + '\n')
                    known[key] = item['response']
    saved = committed(args.data / 'test-observations.jsonl')
    trees = {r['id']: {} for r in rows}
    judges = {}
    for item in saved:
        if item['kind'] == 'solver':
            trees[item['id']][item['path']] = item['response']
        else:
            judges[item['id'], item['path']] = item['response']
    request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.data / 'test-transport-errors.jsonl')
    tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    assert tags[old.MODEL] == old.MODEL_DIGEST and tags[old.JUDGE] == old.JUDGE_DIGEST
    old.request = request
    lock = threading.Lock()
    done = 0
    with (args.data / 'test-observations.jsonl').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def one(full):
            nonlocal done
            row = {k: full[k] for k in ('id', 'question', 'domain', 'subject')}
            row['answer'] = ''
            tree = trees[row['id']]
            for path in PATHS:
                if path in tree:
                    continue
                node = old.call(row, path, tree, judges, False)
                assert node['model_digest'] == old.MODEL_DIGEST
                assert node['backend']['app'] == APP
                with lock:
                    stream.write(json.dumps({'id': row['id'], 'kind': 'solver', 'path': path,
                                             'response': node}) + '\n')
                    stream.flush()
                tree[path] = node
            with lock:
                done += 1
                if done % 24 == 0 or done == len(rows):
                    print(f'Blind fixed-action cache {done}/{len(rows)}; no grades computed', flush=True)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one, rows))
    result = {'questions': len(rows), 'solver_responses': len(rows) * len(PATHS), 'graded': False,
              'contract_sha256': sha(contract),
              'response_bank_sha256': sha(args.data / 'test-observations.jsonl')}
    (args.data / 'fixed-cache-complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    run(p.parse_args())
