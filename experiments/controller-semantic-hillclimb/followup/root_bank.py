"""Separate root-medium trajectories from the ordinary A/B response graph."""

import hashlib

from banks import canonical_review
from root_pairs import PROTOCOL, validate_review
from reasoning_pilot import short_repair
import extended_acquire as old


def view(rows, trees, observations, validate=False):
    by_id = {r['id']: r for r in rows}
    local = {qid: {p: n for p, n in tree.items() if not p.startswith('guided') and not p.endswith('/reviewed')}
             for qid, tree in trees.items()}
    reviews = {}
    known = {(r['id'], r['kind'], r['path']): r['response'] for r in observations}
    if len(known) != len(observations):
        raise ValueError('Duplicate root-medium response')
    for (qid, kind, path), node in known.items():
        if kind == 'solver' and node.get('protocol') == 'ordinary-shared-v1':
            if qid not in by_id or path not in old.PATHS or path.startswith('guided'):
                raise ValueError('Unexpected shared plain path')
            local[qid][path] = node
            continue
        if qid not in by_id or node.get('protocol') != PROTOCOL:
            raise ValueError('Unexpected root-medium response')
        if kind == 'judge' and path == '':
            reviews[qid, path] = canonical_review(node)
        elif kind == 'solver' and path in ('guided', 'guided/repair', 'guided/retry'):
            local[qid][path] = node
        else:
            raise ValueError('Unexpected root-medium path')
    if validate:
        for (qid, kind, path), node in known.items():
            row = {**by_id[qid], 'answer': ''}
            if kind == 'judge':
                validate_review(row, local[qid][''], node)
                assert node['model_digest'] == old.JUDGE_DIGEST
            else:
                if path == 'guided':
                    prompt = short_repair(row, local[qid][''], reviews[qid, ''])
                    cap = 128
                else:
                    prompt = old.solver_prompt(row, path, local[qid], reviews)
                    cap = 256 if path else 128
                logical = path.replace('guided', 'repair')
                seed = int(hashlib.sha256(f'{old.SEED}:{qid}:{logical}'.encode()).hexdigest()[:8], 16) % 2**31
                assert node['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
                assert node['seed'] == seed and node['model_digest'] == old.MODEL_DIGEST
                assert 0 <= node['output_tokens'] <= cap
    return local, reviews
