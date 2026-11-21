"""Keep ordinary and medium-review counterfactual response banks separate."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from semantic_data import records
from late_pairs import merged_records
from medium_late_pairs import PROTOCOL, validate_review, feedback, fingerprint
import extended_acquire as old


def committed(path):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    # JSON strings escape newlines. The last physical newline therefore marks
    # a fully flushed record even while the training collector is appending.
    end = raw.rfind(b'\n') + 1
    return [json.loads(line) for line in raw[:end].splitlines()]


def ordinary(data, split):
    rows, _, original = records(data, split, False)
    _, trees, late = merged_records(data, split, False)
    return rows, trees, original, late


def canonical_review(node):
    if 'finish' in node:
        return node
    if not node.get('calls'):
        raise ValueError('An aggregate review needs its physical phase receipts')
    return {**node, 'finish': node['calls'][-1]['finish']}


def variant_observations(medium, variants=None, variant='compressed'):
    """Pair each paid medium review with one uniformly selected actor protocol."""
    observations = committed(Path(medium) / 'train.jsonl')
    if variant == 'compressed':
        return observations
    if variant not in ('medium-short', 'medium-full') or variants is None:
        raise ValueError('Unknown medium actor protocol')
    leaves = {(r['id'], r['path'] + '/reviewed'): r['response']
              for r in committed(Path(variants) / 'train.jsonl') if r['variant'] == variant}
    result = []
    for item in observations:
        if item['kind'] == 'judge':
            result.append(item)
        else:
            node = leaves.pop((item['id'], item['path']))
            if node['seed'] != item['response']['seed']:
                raise ValueError('Counterfactual compact seeds differ')
            result.append({**item, 'response': node})
    if leaves:
        raise ValueError('Unexpected counterfactual leaf')
    return result


def medium_view(rows, trees, original, observations, validate=False):
    by_id = {r['id']: r for r in rows}
    local = {qid: {p: n for p, n in tree.items() if not p.endswith('/reviewed')}
             for qid, tree in trees.items()}
    reviews = {key: value for key, value in original.items() if key[1] == ''}
    known = {(r['id'], r['kind'], r['path']): r['response'] for r in observations}
    if len(known) != len(observations):
        raise ValueError('Duplicate medium-bank observation')
    for (qid, kind, path), node in known.items():
        if kind == 'solver' and node.get('protocol') == 'ordinary-shared-v1':
            if qid not in by_id or path not in old.PATHS:
                raise ValueError('Unexpected ordinary shared path')
            local[qid][path] = node
            continue
        protocols = (PROTOCOL,) if kind == 'judge' else (PROTOCOL, 'medium-short-late-v1', 'medium-full-late-v1')
        if qid not in by_id or kind not in ('judge', 'solver') or node['protocol'] not in protocols:
            raise ValueError('Unexpected medium-bank observation')
        if kind == 'judge':
            if path not in ('repair', 'retry', 'guided'):
                raise ValueError('A medium review is permitted only after the first continuation')
            reviews[qid, path] = canonical_review(node)
        else:
            if path not in ('repair/reviewed', 'retry/reviewed', 'guided/reviewed'):
                raise ValueError('Unexpected medium-guided leaf')
            local[qid][path] = node
    if validate:
        for (qid, kind, path), node in known.items():
            safe = {**by_id[qid], 'answer': ''}
            if kind == 'judge':
                validate_review(safe, path, local[qid][path], node)
            else:
                if node['protocol'] == 'ordinary-shared-v1':
                    prompt = old.solver_prompt(safe, path, local[qid], reviews)
                    assert node['prompt_sha256'] == fingerprint(prompt.encode())
                    continue
                parent = path.rsplit('/', 1)[0]
                if (qid, parent) not in reviews:
                    raise ValueError('Guided leaf lacks its paid medium review')
                if node['protocol'] == 'medium-short-late-v1':
                    from reasoning_pilot import short_repair
                    prompt = short_repair(safe, local[qid][parent], reviews[qid, parent])
                    cap = 128
                else:
                    packet = (old.feedback(reviews[qid, parent]) if node['protocol'] == 'medium-full-late-v1'
                              else feedback(reviews[qid, parent]))
                    prompt = old.repair_prompt(safe, local[qid][parent], packet)
                    cap = 256
                assert node['prompt_sha256'] == fingerprint(prompt.encode())
                assert node['model_digest'] == old.MODEL_DIGEST and node['output_cap'] == cap
    return local, reviews
