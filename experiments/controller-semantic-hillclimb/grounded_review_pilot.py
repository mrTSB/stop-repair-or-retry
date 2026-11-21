"""Paired training-only factual grounding trial; all baseline receipts stay intact."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import threading

from semantic_data import choose, read, records
from evidence import Retriever
import extended_acquire as old
from review_pilot import inference, prompt as original_prompt
from scoring import correct
from transport import resilient


def grounded_prompt(row, current, context):
    ordinary = original_prompt(row, current, None, 'brief')
    if not context:
        return ordinary
    # The evidence is retrieved using the question stem alone, and is fixed
    # across candidates. No benchmark solutions or correctness labels enter it.
    addition = ('\n\nBACKGROUND EXPOSITION from introductory textbooks; these excerpts may be '
                'incomplete or irrelevant. Check applicability before using them. Derive any '
                'calculation independently.\n' + context)
    marker = '\n\nCURRENT solution:\n'
    if ordinary.count(marker) != 1:
        raise ValueError('Unexpected original review template')
    return ordinary.replace(marker, addition + marker)


def run(args):
    import modal
    rows, trees, judges = records(args.data, 'train', False)
    selected = choose(rows, {'math': args.n // 2, 'chemistry': args.n // 4,
                             'physics': args.n // 4}, 'reasoning-pilot')
    retriever = Retriever(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    contract = {'split': 'train', 'ids': [r['id'] for r in selected],
                'corpus_sha256': retriever.sha256, 'variant': 'grounded-brief-v1',
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'solver_digest': old.MODEL_DIGEST, 'reviewer_digest': old.JUDGE_DIGEST,
                'fresh_data_used': False, 'query': 'question stem only; no options or answers'}
    receipt = args.output.with_suffix('.contract.json')
    if receipt.exists() and json.loads(receipt.read_bytes()) != contract:
        raise ValueError('Changed pilot contract; use a new output file')
    receipt.write_text(json.dumps(contract, indent=2) + '\n')
    request = resilient(modal.Cls.from_name(args.app, 'Ollama')(),
                        args.output.parent / 'grounded-transport-errors.jsonl')
    tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    if tags.get(old.MODEL) != old.MODEL_DIGEST or tags.get(old.JUDGE) != old.JUDGE_DIGEST:
        raise ValueError('Changed frozen model weights')
    with args.output.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        saved = [json.loads(s) for s in stream]
        known = {(r['id'], r['kind']): r for r in saved}
        if len(known) != len(saved):
            raise ValueError('Duplicate pilot receipts')
        lock = threading.Lock()
        finished = 0

        def store(row, kind, node, evidence, reused):
            record = {'id': row['id'], 'kind': kind, 'response': node,
                      'evidence': evidence, 'reused_exact_control': reused}
            with lock:
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                known[row['id'], kind] = record
            return node

        def add(row, kind, prompt, evidence, control=None):
            key = row['id'], kind
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            if key in known:
                node = known[key]['response']
            elif control is not None and control['prompt_sha256'] == digest:
                node = store(row, kind, control, evidence, True)
            else:
                node = inference(request, row, prompt, 'brief', kind == 'review',
                                 '' if kind == 'review' else 'repair')
                node = store(row, kind, node, evidence, False)
            if node['prompt_sha256'] != digest:
                raise ValueError('Changed inference prompt')
            return node

        def one(full):
            nonlocal finished
            row = {**full, 'answer': ''}
            qid = row['id']
            context, evidence = retriever.context(row)
            current = trees[qid]['']
            control_prompt = old.repair_prompt(row, current, old.feedback(judges[qid, '']))
            control = add(row, 'control-repair', control_prompt, [], trees[qid].get('guided'))
            review = add(row, 'review', grounded_prompt(row, current, context), evidence,
                         judges[qid, ''] if not context else None)
            prompt = old.repair_prompt(row, current, old.feedback(review))
            add(row, 'repair', prompt, evidence, control if not context else None)
            with lock:
                finished += 1
                if finished % 8 == 0 or finished == len(selected):
                    print(f'Grounded training pilot {finished}/{len(selected)}', flush=True)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one, selected))
    result = {'n': len(selected), 'split': 'training only', 'fresh_data_used': False,
              'corpus_sha256': retriever.sha256, 'methods': {}, 'by_subject': {}}
    methods = {
        'standard-review': lambda r: judges[r['id'], ''],
        'standard-guided-repair': lambda r: known[r['id'], 'control-repair']['response'],
        'grounded-review': lambda r: known[r['id'], 'review']['response'],
        'grounded-guided-repair': lambda r: known[r['id'], 'repair']['response'],
    }

    def score(group):
        values = {}
        for name, get in methods.items():
            nodes = [get(r) for r in group]
            values[name] = {'n': len(group),
                            'correct': sum(correct(n, r['answer']) for r, n in zip(group, nodes)),
                            'mean_tokens': sum(n['input_tokens'] + n['output_tokens'] for n in nodes) / len(nodes)}
        return values

    result['methods'] = score(selected)
    for subject in ('math', 'chemistry', 'physics'):
        result['by_subject'][subject] = score([r for r in selected if r['subject'] == subject])
    result['new_inference_nodes'] = sum(not r['reused_exact_control'] for r in known.values())
    args.output.with_suffix('.summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n', type=int, default=96)
    parser.add_argument('--app', default='srr-grounded-solver-20261120')
    args = parser.parse_args()
    if args.n % 4:
        parser.error('Use a multiple of four')
    run(args)
