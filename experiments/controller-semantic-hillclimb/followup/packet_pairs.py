"""Complete every training continuation for the matched full-review repair format."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import threading

from banks import ordinary, committed
from full_packet_pilot import prompt, PROTOCOL, sha
from packet_bank import view, PLAIN_PROTOCOL
from medium_late_pairs import APP
from transfer_variants import actor
from scoring import correct
from transport import resilient
import extended_acquire as old


def run(args):
    import modal
    rows, trees, original, _ = ordinary(args.data, 'train')
    chosen = json.loads((args.root / 'contract.json').read_bytes())['selection']
    pilot = committed(args.pilot / 'train.jsonl')
    if len(chosen) != 192 or len(pilot) != 192 or {r['id'] for r in chosen} != {r['id'] for r in pilot}:
        raise ValueError('Complete the entire matched192question pilot first')
    by_id = {r['id']: r for r in rows}
    args.output.mkdir(parents=True, exist_ok=True)
    contract = {'selection': chosen, 'questions': 192, 'expected_responses': 960,
                'source_sha256': sha(__file__), 'pilot_sha256': sha(args.pilot / 'train.jsonl'),
                'prompt_source_sha256': sha(Path(__file__).with_name('full_packet_pilot.py')),
                'solver_digest': old.MODEL_DIGEST, 'judge_digest': old.JUDGE_DIGEST,
                'additional_judge_calls': 0, 'fresh_data_used': False,
                'rule': 'Every matched root question; exact existing brief review, full-packet guided response, both parents reuse and a fresh guided-child retry. No outcome filtering.'}
    receipt = args.output / 'contract.json'
    if receipt.exists() and json.loads(receipt.read_bytes()) != contract:
        raise ValueError('Changed packet-pair protocol')
    receipt.write_text(json.dumps(contract, indent=2) + '\n')
    request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
    tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    assert tags[old.MODEL] == old.MODEL_DIGEST and tags[old.JUDGE] == old.JUDGE_DIGEST
    old.request = request
    path = args.output / 'train.jsonl'
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = committed(path)
        known = {(r['id'], r['kind'], r['path']): r['response'] for r in saved}
        if len(known) != len(saved):
            raise ValueError('Duplicate stored response')
        lock = threading.Lock()
        done = 0

        def append(qid, kind, p, node):
            stream.write(json.dumps({'id': qid, 'kind': kind, 'path': p, 'response': node}) + '\n')
            stream.flush()
            known[qid, kind, p] = node

        for item in chosen:
            qid = item['id']
            key = qid, 'judge', ''
            if key not in known:
                append(qid, 'judge', '', original[qid, ''])
            elif known[key] != original[qid, '']:
                raise ValueError('Changed reused ordinary review')
        for item in pilot:
            key = item['id'], 'solver', 'guided'
            if key not in known:
                append(item['id'], 'solver', 'guided', item['response'])
            elif known[key] != item['response']:
                raise ValueError('Changed imported guided response')

        def one(item):
            nonlocal done
            qid = item['id']; row = {**by_id[qid], 'answer': ''}
            local = {**trees[qid], 'guided': known[qid, 'solver', 'guided']}
            for p in ('guided/retry', 'guided/reuse', 'retry/reuse'):
                key = qid, 'solver', p
                if key not in known:
                    if p == 'guided/retry':
                        node = old.call(row, p, local, original, False)
                        node['protocol'] = PLAIN_PROTOCOL
                    else:
                        parent = p.rsplit('/', 1)[0]
                        text = prompt(row, local[parent], original[qid, ''])
                        logical = p.replace('guided', 'repair').replace('reuse', 'repair')
                        seed = int(hashlib.sha256(f'{old.SEED}:{qid}:{logical}'.encode()).hexdigest()[:8], 16) % 2**31
                        node = actor(request, row, text, seed, 128)
                        node['protocol'] = PROTOCOL
                    with lock:
                        append(qid, 'solver', p, node)
            with lock:
                done += 1
                if done % 24 == 0 or done == 192:
                    print(f'Full-packet paired training continuations {done}/192', flush=True)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one, chosen))
    observations = committed(path)
    local, _ = view(rows, trees, original, observations, validate=True)
    if len(observations) != 960:
        raise ValueError('Incomplete paired training bank')
    summary = {'questions': 192, 'responses': 960, 'additional_judge_calls': 0,
               'fresh_data_used': False, 'bank_sha256': sha(path), 'paths': {}}
    for p in ('guided', 'guided/retry', 'guided/reuse', 'retry/reuse'):
        nodes = [local[r['id']][p] for r in chosen]
        summary['paths'][p] = {'correct': sum(correct(n, by_id[r['id']]['answer']) for r, n in zip(chosen, nodes)),
                               'mean_solver_tokens': sum(n['input_tokens'] + n['output_tokens'] for n in nodes) / 192}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'root', 'pilot', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
