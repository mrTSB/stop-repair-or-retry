"""Collect every training counterfactual for reusing an already-paid root review."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import threading

from banks import ordinary, committed
from root_bank import view
from reasoning_pilot import short_repair
from transfer_variants import actor
from medium_late_pairs import APP
from transport import resilient
from scoring import correct
import extended_acquire as old

PROTOCOL = 'root-mentor-reuse-v1'


def run(args):
    import modal
    rows, trees, _, _ = ordinary(args.data, 'train')
    chosen = json.loads((args.root / 'contract.json').read_bytes())['selection']
    bank = committed(args.root / 'train.jsonl')
    if len(bank) != 768 or len(chosen) != 192:
        raise ValueError('Finish all192original root pairs first')
    local, judges = view(rows, trees, bank, validate=True)
    by_id = {r['id']: r for r in rows}
    args.output.mkdir(parents=True, exist_ok=True)
    contract = {'questions': 192, 'parents': ['guided', 'retry'], 'responses': 384,
                'root_bank_sha256': hashlib.sha256((args.root / 'train.jsonl').read_bytes()).hexdigest(),
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'rule': 'Every selected question and both parents, irrespective correctness. Reuse the existing paid root proposal/check; no new LLM review.',
                'solver_digest': old.MODEL_DIGEST, 'cap': 128, 'fresh_data_used': False}
    path = args.output / 'contract.json'
    if path.exists() and json.loads(path.read_bytes()) != contract:
        raise ValueError('Changed reuse-pair protocol')
    path.write_text(json.dumps(contract, indent=2) + '\n')
    request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
    tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    assert tags[old.MODEL] == old.MODEL_DIGEST
    with (args.output / 'train.jsonl').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = committed(args.output / 'train.jsonl')
        known = {(r['id'], r['path']): r['response'] for r in saved}
        assert len(known) == len(saved)
        lock = threading.Lock()
        completed = 0

        def one(item):
            nonlocal completed
            qid = item['id']
            row = {**by_id[qid], 'answer': ''}
            for parent in ('guided', 'retry'):
                leaf = parent + '/reuse'
                prompt = short_repair(row, local[qid][parent], judges[qid, ''])
                logical = 'repair/repair' if parent == 'guided' else 'retry/repair'
                seed = int(hashlib.sha256(f'{old.SEED}:{qid}:{logical}'.encode()).hexdigest()[:8], 16) % 2**31
                if (qid, leaf) not in known:
                    node = actor(request, row, prompt, seed, 128)
                    node['protocol'] = PROTOCOL
                    with lock:
                        stream.write(json.dumps({'id': qid, 'kind': 'solver', 'path': leaf, 'response': node}) + '\n')
                        stream.flush()
                        known[qid, leaf] = node
                node = known[qid, leaf]
                assert node['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest() and node['seed'] == seed
            with lock:
                completed += 1
                if completed % 24 == 0 or completed == 192:
                    print(f'Paid-review reuse training pairs {completed}/192', flush=True)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one, chosen))
    result = {'questions': 192, 'responses': len(known), 'fresh_data_used': False, 'parents': {}}
    for parent in ('guided', 'retry'):
        nodes = [known[r['id'], parent + '/reuse'] for r in chosen]
        controls = [(r, local[r['id']][parent + '/repair']) for r in chosen if parent + '/repair' in local[r['id']]]
        result['parents'][parent] = {'reuse_correct': sum(correct(n, by_id[r['id']]['answer']) for r, n in zip(chosen, nodes)),
                                     'mean_reuse_tokens': sum(n['input_tokens'] + n['output_tokens'] for n in nodes) / 192,
                                     'ordinary_control_available': len(controls),
                                     'ordinary_control_correct': sum(correct(n, by_id[r['id']]['answer']) for r, n in controls)}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
