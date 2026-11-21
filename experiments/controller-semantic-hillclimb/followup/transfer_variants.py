"""Reuse paid training reviews to test compact-solver transfer, never retrying by correctness."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import ordinary, committed
from medium_late_pairs import APP
from reasoning_pilot import short_repair
import extended_acquire as old
from scoring import correct
from transport import resilient

VARIANTS = ('standard-short', 'medium-short', 'medium-full')


def actor(request, row, prompt, seed, cap):
    started = time.monotonic()
    result = request('/api/generate', {'model': old.MODEL, 'prompt': prompt, 'raw': True,
        'stream': False, 'logprobs': True, 'keep_alive': '30m', 'options': {
            'temperature': .65, 'seed': seed, 'num_predict': cap, 'num_ctx': 4096}})
    assert result.get('done') and result.get('model') == old.MODEL and result.get('prompt_eval_count')
    text = result['response']
    if text and not result.get('logprobs'):
        raise ValueError('Missing compact token probabilities')
    return {'model': old.MODEL, 'model_digest': old.MODEL_DIGEST, 'text': text,
            'answer': old.parse_answer(row['domain'], text),
            'confidence': old.confidence(text, result.get('logprobs', []), row['domain']),
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'seed': seed,
            'temperature': .65, 'output_cap': cap, 'context_length': 4096,
            'input_tokens': int(result['prompt_eval_count']), 'output_tokens': int(result.get('eval_count', 0)),
            'finish': result['done_reason'], 'latency_seconds': time.monotonic() - started,
            'backend': result.get('srr_backend')}


def collect(args):
    import modal
    rows, trees, _, standard = ordinary(args.data, 'train')
    by_id = {r['id']: r for r in rows}
    contract = json.loads((args.medium / 'contract.json').read_bytes())
    medium = {(r['id'], r['kind'], r['path']): r['response'] for r in committed(args.medium / 'train.jsonl')}
    if len(medium) != 2 * len(contract['selection']):
        raise ValueError('Finish every original training pair before transfer variants')
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'medium_bank_sha256': hashlib.sha256((args.medium / 'train.jsonl').read_bytes()).hexdigest(),
              'variants': VARIANTS, 'seed_rule': 'Same integer compact seed as the matched original medium repair.',
              'selection_rule': 'Every one of the prespecified192training pairs; no correctness-based subset.',
              'fresh_data_used': False}
    path = args.output / 'contract.json'
    if path.exists():
        assert json.loads(path.read_bytes()) == {**recipe, 'variants': list(VARIANTS)}
    path.write_text(json.dumps(recipe, indent=2) + '\n')
    request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
    with (args.output / 'train.jsonl').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        saved = [json.loads(s) for s in stream]
        known = {(r['id'], r['path'], r['variant']): r['response'] for r in saved}
        assert len(known) == len(saved)
        lock = threading.Lock()
        done = 0

        def one(item):
            nonlocal done
            qid, path = item['id'], item['path']
            row = {**by_id[qid], 'answer': ''}
            current = trees[qid][path]
            deep = medium[qid, 'judge', path]
            seed = medium[qid, 'solver', path + '/reviewed']['seed']
            for variant in VARIANTS:
                mentor = standard[qid, path] if variant == 'standard-short' else deep
                prompt = (old.repair_prompt(row, current, old.feedback(mentor)) if variant == 'medium-full'
                          else short_repair(row, current, mentor))
                cap = 256 if variant == 'medium-full' else 128
                key = qid, path, variant
                if key not in known:
                    node = actor(request, row, prompt, seed, cap)
                    node['protocol'] = variant + '-late-v1'
                    with lock:
                        stream.write(json.dumps({'id': qid, 'path': path, 'variant': variant, 'response': node}) + '\n')
                        stream.flush()
                        known[key] = node
                node = known[key]
                assert (node['prompt_sha256'], node['seed'], node['output_cap']) == (
                    hashlib.sha256(prompt.encode()).hexdigest(), seed, cap)
            with lock:
                done += 1
                if done % 16 == 0 or done == len(contract['selection']):
                    print(f'Complete transfer variants {done}/{len(contract["selection"])}', flush=True)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one, contract['selection']))
    summary = {'n': len(contract['selection']), 'split': 'training only', 'fresh_data_used': False,
               'methods': {}}
    for variant in VARIANTS:
        nodes = [known[r['id'], r['path'], variant] for r in contract['selection']]
        summary['methods'][variant] = {'correct': sum(correct(node, by_id[r['id']]['answer'])
                                                      for r, node in zip(contract['selection'], nodes)),
                                       'mean_solver_tokens': sum(n['input_tokens'] + n['output_tokens'] for n in nodes) / len(nodes)}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--medium', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    collect(parser.parse_args())
