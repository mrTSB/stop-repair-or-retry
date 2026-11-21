"""Matched training-only reasoning-dose pairs using the same frozen LLM reviewer."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import ordinary, committed
from full_packet_pilot import sha
from long_review_pilot import review
from reasoning_pilot import short_repair, inference
from scoring import correct
from transport import resilient
import extended_acquire as old

APP = 'srr-followup-solver-20261120'


def run(args):
    import modal
    rows, trees, ordinary_reviews, _ = ordinary(args.data, 'train')
    selected = json.loads((args.selection / 'contract.json').read_bytes())['selection']
    by_id = {r['id']: r for r in rows}
    if len(selected) != 192 or not {r['id'] for r in selected} <= set(by_id):
        raise ValueError('Use the entire matched192training cohort')
    contract = {'source_sha256': sha(__file__), 'selection_sha256': sha(args.selection / 'contract.json'),
                'training_sha256': sha(args.data / 'train-observations.jsonl'), 'n': 192,
                'think_caps': [256, 512], 'answer_cap': 160, 'solver_cap': 128,
                'solver_digest': old.MODEL_DIGEST, 'reviewer_digest': old.JUDGE_DIGEST,
                'selection': 'All192preexisting matched training questions, every reasoning dose. No outcome filtering.',
                'fresh_data_used': False, 'max_cloud_gpus': 2}
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = args.output / 'contract.json'
    if recipe.exists() and json.loads(recipe.read_bytes()) != contract:
        raise ValueError('Changed reasoning-dose contract')
    recipe.write_text(json.dumps(contract, indent=2) + '\n')
    request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
    actual = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    assert actual[old.MODEL] == old.MODEL_DIGEST and actual[old.JUDGE] == old.JUDGE_DIGEST
    target = args.output / 'train.jsonl'
    known = {(r['id'], r['dose'], r['kind']): r['response'] for r in committed(target)}
    assert len(known) == len(committed(target))
    lock = threading.Lock()
    done = 0
    with target.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def store(qid, dose, kind, node):
            node = {**node, 'protocol': f'budget-root-{dose}-v1'}
            with lock:
                stream.write(json.dumps({'id': qid, 'dose': dose, 'kind': kind, 'response': node}) + '\n')
                stream.flush()
                known[qid, dose, kind] = node
            return node

        def one(item):
            nonlocal done
            qid = item['id']
            row = {**by_id[qid], 'answer': ''}
            current = trees[qid]['']
            for dose in (256, 512):
                teacher = known.get((qid, dose, 'review'))
                if teacher is None:
                    teacher = store(qid, dose, 'review', review(request, row, current, dose, 160))
                prompt = short_repair(row, current, teacher)
                if (qid, dose, 'guided') not in known:
                    store(qid, dose, 'guided', inference(request, row, prompt, False, 128))
                assert known[qid, dose, 'guided']['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
            with lock:
                done += 1
                if done % 12 == 0 or done == 192:
                    print(f'Paired reasoning-dose training questions {done}/192', flush=True)

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(one, selected))
    result = {'n': 192, 'split': 'training only', 'fresh_data_used': False,
              'bank_sha256': sha(target), 'methods': {}}
    for dose in (256, 512):
        teachers = [known[r['id'], dose, 'review'] for r in selected]
        students = [known[r['id'], dose, 'guided'] for r in selected]
        result['methods'][str(dose)] = {
            'teacher_correct': sum(correct(n, by_id[r['id']]['answer']) for r, n in zip(selected, teachers)),
            'student_correct': sum(correct(n, by_id[r['id']]['answer']) for r, n in zip(selected, students)),
            'mean_teacher_tokens': sum(n['input_tokens'] + n['output_tokens'] for n in teachers) / 192,
            'mean_student_tokens': sum(n['input_tokens'] + n['output_tokens'] for n in students) / 192,
            'physical_teacher_calls': sum(n['physical_api_calls'] for n in teachers),
            'invalid_teachers': sum(not n['parse_valid'] or not n['answer'] for n in teachers)}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'selection', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
