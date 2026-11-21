"""Replay the frozen912policies without references, acquiring only missing paid responses."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import threading

from confirmation_common import evidence, models, run_policy, read, sha, verify_fixed_prefix, verify_special_banks
from banks import committed, canonical_review
from medium_late_pairs import APP, PROTOCOL, scoped
from reasoning_pilot import short_repair
from long_review_pilot import review as deeper_review
from transfer_variants import actor
from root_pairs import PROTOCOL as ROOT_PROTOCOL
from reuse_pairs import PROTOCOL as REUSE_PROTOCOL
from reasoning_pilot import inference as root_actor
from late_pairs import judge_prompt, sanitized
from review_pilot import inference
from difficulty_pomdp import MissingObservation
from transport import resilient
import extended_acquire as old


def run(args):
    import modal
    frozen = json.loads((args.output / 'frozen.json').read_bytes())
    if frozen['evidence'] != evidence(args.data):
        raise ValueError('Changed policy sources or development evidence')
    verify_fixed_prefix(frozen, args.data)
    verify_special_banks(frozen, args.medium)
    audit = json.loads((args.output / 'development-audit.json').read_bytes())
    if not audit['passed'] or audit['freeze_sha256'] != sha(args.output / 'frozen.json'):
        raise ValueError('Complete independent development checks first')
    if audit['training_only_refitted_outcomes'] != 480 or audit['independent_encoder_regenerations'] <= 0:
        raise ValueError('Reconstruct both planners from training data and regenerate observations before fresh replay')
    fixed = json.loads((args.data / 'fixed-cache-complete.json').read_bytes())
    if fixed['questions'] != 912 or fixed['contract_sha256'] != frozen['fixed_cache_contract_sha256']:
        raise ValueError('Finish every prespecified fixed baseline response first')
    rows = read(args.data / 'splits/test-questions.jsonl')
    if len(rows) != 912:
        raise ValueError('Changed fresh panel')
    planners = models(frozen, args.output)
    root_protocol = frozen['selected']['C']['protocol']
    root_mode = root_protocol in ('root-medium', 'mentor-reuse')
    reuse_mode = root_protocol == 'mentor-reuse'
    identity = {'freeze_sha256': sha(args.output / 'frozen.json'), 'questions': 912,
                'rule': 'All references removed; no test correctness or question selection is consulted.'}
    metadata = args.data / 'test-collection.json'
    if metadata.exists():
        previous = json.loads(metadata.read_bytes())
        if any(previous[k] != v for k, v in identity.items()):
            raise ValueError('This panel belongs to a different controller freeze')
        completed = set(previous['completed_ids'])
    else:
        completed = set()
    request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.data / 'test-transport-errors.jsonl')
    tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    assert tags[old.MODEL] == old.MODEL_DIGEST and tags[old.JUDGE] == old.JUDGE_DIGEST
    old.request = request
    paths = {'base': args.data / 'test-observations.jsonl',
             'late': args.data / 'test-late-observations.jsonl', 'medium': args.medium / 'test.jsonl'}
    if reuse_mode:
        paths['reuse'] = Path(frozen['reuse_bank']['directory']) / 'test.jsonl'
    paths['medium'].parent.mkdir(parents=True, exist_ok=True)
    cache_path = args.output / 'test-neural-cache.jsonl'
    recorded = {(r['family'], tuple(r['key'])) for r in committed(cache_path)}
    lock = threading.Lock()
    known, streams = {}, {}
    try:
        for protocol, path in paths.items():
            stream = path.open('a+')
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            streams[protocol] = stream
            items = committed(path)
            known[protocol] = {(r['id'], r['kind'], r['path']): r['response'] for r in items}
            if len(known[protocol]) != len(items):
                raise ValueError('Duplicate response')
        ids = {r['id'] for r in rows}
        if not completed <= ids or any(qid not in ids for values in known.values() for qid, _, _ in values):
            raise ValueError('Unexpected question ID')

        def append(row, storage, kind, path, node):
            key = row['id'], kind, path
            with lock:
                if key in known[storage]:
                    raise ValueError('Duplicate concurrent response')
                streams[storage].write(json.dumps({'id': row['id'], 'kind': kind, 'path': path,
                                                  'response': node}) + '\n')
                streams[storage].flush()
                known[storage][key] = node

        def one(full):
            row = {k: full[k] for k in ('id', 'question', 'domain', 'subject')}
            row['answer'] = ''
            with lock:
                if row['id'] in completed:
                    return
                local = {protocol: {(kind, p): node for (qid, kind, p), node in values.items() if qid == row['id']}
                         for protocol, values in known.items()}
            tree = {p: n for (kind, p), n in local['base'].items() if kind == 'solver'}
            tree.update({p: n for (kind, p), n in local['late'].items() if kind == 'solver'})
            original = {(row['id'], p): n for (kind, p), n in local['base'].items() if kind == 'judge'}
            late = {k: v for k, v in original.items() if k[1] == ''}
            late.update({(row['id'], p): n for (kind, p), n in local['late'].items() if kind == 'judge'})
            deeper = {p: n for p, n in tree.items()
                      if not p.endswith('/reviewed') and not (root_mode and p.startswith('guided'))}
            deep_reviews = {} if root_mode else {k: v for k, v in original.items() if k[1] == ''}
            for (kind, p), node in local['medium'].items():
                if kind == 'solver':
                    deeper[p] = node
                else:
                    deep_reviews[row['id'], p] = canonical_review(node)
            if reuse_mode:
                for (kind, p), node in local['reuse'].items():
                    assert kind == 'solver' and p in ('guided/reuse', 'retry/reuse')
                    deeper[p] = node

            def acquire(kind, path, protocol):
                special = protocol in ('medium', 'root-medium', 'mentor-reuse')
                current_root = protocol in ('root-medium', 'mentor-reuse')
                current = deeper if special else tree
                if kind == 'judge' and path not in current:
                    acquire('solver', path, protocol)
                if kind == 'solver' and path:
                    parent = path.rsplit('/', 1)[0] if '/' in path else ''
                    if parent not in current:
                        acquire('solver', parent, protocol)
                    if path == 'guided':
                        reviews = deep_reviews if current_root else original
                        if (row['id'], '') not in reviews:
                            acquire('judge', '', protocol if current_root else 'original')
                    if path.endswith('/reuse') and (row['id'], '') not in deep_reviews:
                        acquire('judge', '', protocol)
                    if path.endswith('/reviewed'):
                        reviews = deep_reviews if protocol == 'medium' else late
                        if (row['id'], parent) not in reviews:
                            acquire('judge', parent, protocol)
                if current_root and kind == 'judge':
                    if path != '':
                        raise ValueError('The root hybrid cannot acquire a late LLM review')
                    node = deeper_review(request, row, deeper[''], 1024, 512)
                    node['protocol'] = ROOT_PROTOCOL
                    node['backend'] = node['calls'][0]['backend']
                    deep_reviews[row['id'], ''] = canonical_review(node)
                    storage = 'medium'
                elif protocol == 'mentor-reuse' and kind == 'solver' and path.endswith('/reuse'):
                    parent = path.rsplit('/', 1)[0]
                    if parent not in ('guided', 'retry'):
                        raise ValueError('Unexpected mentor-reuse parent')
                    formatted = short_repair(row, deeper[parent], deep_reviews[row['id'], ''])
                    logical = 'repair/repair' if parent == 'guided' else 'retry/repair'
                    seed = int(hashlib.sha256(f'{old.SEED}:{row["id"]}:{logical}'.encode()).hexdigest()[:8], 16) % 2**31
                    node = actor(request, row, formatted, seed, 128)
                    node['protocol'] = REUSE_PROTOCOL
                    deeper[path] = node
                    storage = 'reuse'
                elif current_root and kind == 'solver' and path.startswith('guided'):
                    if path == 'guided':
                        formatted = short_repair(row, deeper[''], deep_reviews[row['id'], ''])
                        node = root_actor(request, row, formatted, False, 128)
                    elif path in ('guided/repair', 'guided/retry'):
                        node = old.call(row, path, deeper, deep_reviews, False)
                    else:
                        raise ValueError('Unexpected root-hybrid continuation')
                    node['protocol'] = ROOT_PROTOCOL
                    deeper[path] = node
                    storage = 'medium'
                elif protocol == 'medium' and path and kind == 'judge':
                    if path not in ('repair', 'retry', 'guided'):
                        raise ValueError('Unexpected deeper review checkpoint')
                    node = deeper_review(request, scoped(row, path), deeper[path], 1024, 512)
                    node['protocol'] = PROTOCOL
                    node['backend'] = node['calls'][0]['backend']
                    deep_reviews[row['id'], path] = canonical_review(node)
                    storage = 'medium'
                elif protocol == 'medium' and path.endswith('/reviewed') and kind == 'solver':
                    parent = path.rsplit('/', 1)[0]
                    formatted = short_repair(row, deeper[parent], deep_reviews[row['id'], parent])
                    seed = int(hashlib.sha256(f'{old.SEED}:{row["id"]}:{parent}/repair'.encode()).hexdigest()[:8], 16) % 2**31
                    node = actor(request, row, formatted, seed, 128)
                    node['protocol'] = 'medium-short-late-v1'
                    deeper[path] = node
                    storage = 'medium'
                elif kind == 'judge' and path and protocol == 'late':
                    node = sanitized(inference(request, row, judge_prompt(row, path, tree), 'brief', True, path))
                    node['protocol'] = 'late-reviewed-repair-v1'
                    late[row['id'], path] = node
                    storage = 'late'
                elif kind == 'solver' and path.endswith('/reviewed'):
                    parent = path.rsplit('/', 1)[0]
                    formatted = old.repair_prompt(row, tree[parent], old.feedback(late[row['id'], parent]))
                    node = inference(request, row, formatted, 'brief', False, parent + '/repair')
                    node['protocol'] = 'late-reviewed-repair-v1'
                    tree[path] = node
                    storage = 'late'
                else:
                    if (kind == 'solver' and path not in old.PATHS) or (kind == 'judge' and path not in old.CHECKPOINTS):
                        raise ValueError('Unexpected ordinary response request')
                    node = old.call(row, path, tree, original, kind == 'judge')
                    storage = 'base'
                    if kind == 'solver':
                        tree[path] = node
                        if not (root_mode and path.startswith('guided')):
                            deeper[path] = node
                    else:
                        original[row['id'], path] = node
                        if path == '':
                            late[row['id'], path] = node
                            if not root_mode:
                                deep_reviews[row['id'], path] = node
                assert node['backend']['app'] == APP
                append(row, storage, kind, path, node)

            for family in ('A', 'B', 'C'):
                config = frozen['selected'][family]
                while True:
                    try:
                        run_policy(family, row, tree, original, late, deeper, deep_reviews,
                                   planners.get(family), config)
                        break
                    except MissingObservation as missing:
                        if family == 'A' and missing.kind == 'judge':
                            raise ValueError('POMDP A attempted an LLM call')
                        acquire(missing.kind, missing.path, config['protocol'])
                    except KeyError as error:
                        key = error.args[0]
                        if isinstance(key, tuple) and len(key) == 2 and key[0] == row['id']:
                            acquire('judge', key[1], config['protocol'])
                        elif isinstance(key, str) and key in old.PATHS:
                            acquire('solver', key, config['protocol'])
                        else:
                            raise
            with lock:
                with cache_path.open('a') as stream:
                    for family, model in planners.items():
                        recipe_sha = sha(args.output / frozen['models'][family]['observer_directory'] / 'recipe.json')
                        for key, values in list(model.classifier._cache.items()):
                            if key[1] != row['id'] or (family, key) in recorded:
                                continue
                            stream.write(json.dumps({'family': family, 'key': key, 'recipe_sha256': recipe_sha,
                                                     'state': values[0].tolist(), 'plain': values[1].tolist(),
                                                     'guided': values[2].tolist(), 'tokens': values[3]}) + '\n')
                            recorded.add((family, key))
                completed.add(row['id'])
                receipt = {**identity, 'completed': len(completed), 'completed_ids': sorted(completed),
                           'neural_cache_sha256': sha(cache_path),
                           **{name + '_sha256': sha(path) for name, path in paths.items()}}
                temporary = metadata.with_suffix('.tmp')
                temporary.write_text(json.dumps(receipt, indent=2) + '\n')
                temporary.replace(metadata)
                if len(completed) % 24 == 0 or len(completed) == 912:
                    print(f'Frozen blind controller replay {len(completed)}/912; no grades inspected', flush=True)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one, rows))
    finally:
        for stream in streams.values():
            stream.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'medium', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
