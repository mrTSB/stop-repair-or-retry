"""Isolate brief-review, full-packet guided paths from the unchanged references."""

import hashlib

from full_packet_pilot import prompt, PROTOCOL
from banks import canonical_review
import extended_acquire as old

PLAIN_PROTOCOL = 'full-packet-plain-retry-v1'


def view(rows, trees, original, observations, validate=False):
    by_id = {r['id']: r for r in rows}
    local = {qid: {p: n for p, n in tree.items() if not p.startswith('guided') and not p.endswith('/reviewed')}
             for qid, tree in trees.items()}
    reviews = {}
    known = {}
    for item in observations:
        key = item['id'], item['kind'], item['path']
        if key in known or item['id'] not in by_id:
            raise ValueError('Duplicate or unexpected full-packet observation')
        known[key] = item['response']
        if item['kind'] == 'judge':
            if item['path'] or item['response'] != original[item['id'], '']:
                raise ValueError('Full-packet C must use the exact ordinary brief root review')
            reviews[item['id'], ''] = canonical_review(item['response'])
    for (qid, kind, path), node in sorted(known.items(), key=lambda p: p[0][2].count('/')):
        if kind == 'judge':
            continue
        shared = node.get('protocol') == 'ordinary-shared-v1'
        if shared:
            if kind != 'solver' or path not in old.PATHS or path.startswith('guided'):
                raise ValueError('Unexpected shared ordinary response')
            if validate:
                expected = old.solver_prompt({**by_id[qid], 'answer': ''}, path, local[qid], reviews)
                assert node['prompt_sha256'] == hashlib.sha256(expected.encode()).hexdigest()
            local[qid][path] = node
            continue
        if kind != 'solver' or path not in ('guided', 'guided/retry', 'guided/reuse', 'retry/reuse'):
            raise ValueError('Unexpected full-packet solver path')
        parent = path.rsplit('/', 1)[0] if '/' in path else ''
        if parent not in local[qid] or (qid, '') not in reviews:
            raise ValueError('Full-packet repair lacks its paid prefix')
        if validate:
            safe = {**by_id[qid], 'answer': ''}
            logical = path.replace('guided', 'repair').replace('reuse', 'repair')
            seed = int(hashlib.sha256(f'{old.SEED}:{qid}:{logical}'.encode()).hexdigest()[:8], 16) % 2**31
            if path == 'guided/retry':
                expected = old.solver_prompt(safe, path, local[qid], reviews)
                assert node['protocol'] == PLAIN_PROTOCOL
            else:
                expected = prompt(safe, local[qid][parent], reviews[qid, ''])
                assert node['protocol'] == PROTOCOL and node['output_cap'] == 128
            assert node['prompt_sha256'] == hashlib.sha256(expected.encode()).hexdigest()
            assert node['seed'] == seed and node['model_digest'] == old.MODEL_DIGEST
        local[qid][path] = node
    return local, reviews
