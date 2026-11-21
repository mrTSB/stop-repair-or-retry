"""Matched training-only test of answer-first repair retaining the full brief review."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import threading

from banks import ordinary, committed
from medium_late_pairs import APP
from reasoning_pilot import inference
from scoring import correct
from transport import resilient
import extended_acquire as old

PROTOCOL = 'full-packet-answer-first-v1'


def prompt(row, current, review):
    unit = 'one option letter' if row['domain'] == 'science' else 'a number'
    return old.chat('Correct the CURRENT solution using the independent review. Check that the proposed answer '
                    'answers the original question. Write a line "Final: X" FIRST, where X is ' + unit +
                    ', then at most 35 words justifying it.\n\n' + row['question'].removesuffix(old.SCIENCE_SUFFIX) +
                    '\n\nCURRENT solution:\n' + current['text'] + '\n\nREVIEW:\n' + old.feedback(review))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    import modal
    rows, trees, original, _ = ordinary(args.data, 'train')
    chosen = json.loads((args.root / 'contract.json').read_bytes())['selection']
    by_id = {r['id']: r for r in rows}
    if len(chosen) != 192 or not {r['id'] for r in chosen} <= set(by_id):
        raise ValueError('Use exactly the matched192training questions, never fresh data')
    args.output.mkdir(parents=True, exist_ok=True)
    controls = {(r['id'], r['path']): r['response'] for r in committed(args.root / 'controls.jsonl')}
    contract = {'questions': 192, 'source_sha256': sha(__file__),
                'selection_sha256': sha(args.root / 'contract.json'),
                'ordinary_training_sha256': sha(args.data / 'train-observations.jsonl'),
                'solver_digest': old.MODEL_DIGEST, 'output_cap': 128,
                'additional_judge_calls': 0, 'fresh_data_used': False,
                'comparison': 'Every matched question; original full guided repair versus full-packet answer-first repair. No outcome filtering.'}
    receipt = args.output / 'contract.json'
    if receipt.exists() and json.loads(receipt.read_bytes()) != contract:
        raise ValueError('Changed pilot; use a new output directory')
    receipt.write_text(json.dumps(contract, indent=2) + '\n')
    request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
    tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    assert tags[old.MODEL] == old.MODEL_DIGEST
    path = args.output / 'train.jsonl'
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = committed(path)
        known = {r['id']: r['response'] for r in saved}
        if len(known) != len(saved):
            raise ValueError('Duplicate response')
        lock = threading.Lock()
        done = 0

        def one(item):
            nonlocal done
            row = {**by_id[item['id']], 'answer': ''}
            text = prompt(row, trees[row['id']][''], original[row['id'], ''])
            if row['id'] not in known:
                node = inference(request, row, text, False, 128)
                node['protocol'] = PROTOCOL
                with lock:
                    stream.write(json.dumps({'id': row['id'], 'kind': 'solver', 'path': 'guided', 'response': node}) + '\n')
                    stream.flush()
                    known[row['id']] = node
            assert known[row['id']]['prompt_sha256'] == hashlib.sha256(text.encode()).hexdigest()
            with lock:
                done += 1
                if done % 24 == 0 or done == 192:
                    print(f'Full-packet answer-first training repairs {done}/192', flush=True)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one, chosen))
    original_nodes = [trees[r['id']].get('guided', controls.get((r['id'], 'guided'))) for r in chosen]
    if any(n is None for n in original_nodes):
        raise ValueError('Complete every original guided control first')
    new_nodes = [known[r['id']] for r in chosen]
    results = {'questions': 192, 'fresh_data_used': False, 'additional_judge_calls': 0, 'methods': {}}
    for name, nodes in (('original-full', original_nodes), ('full-packet-answer-first', new_nodes)):
        results['methods'][name] = {'correct': sum(correct(n, by_id[r['id']]['answer']) for r, n in zip(chosen, nodes)),
                                  'mean_solver_tokens': sum(n['input_tokens'] + n['output_tokens'] for n in nodes) / 192}
    old_ok = [correct(n, by_id[r['id']]['answer']) for r, n in zip(chosen, original_nodes)]
    new_ok = [correct(n, by_id[r['id']]['answer']) for r, n in zip(chosen, new_nodes)]
    results['paired_wins'] = sum(b and not a for a, b in zip(old_ok, new_ok))
    results['paired_losses'] = sum(a and not b for a, b in zip(old_ok, new_ok))
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
