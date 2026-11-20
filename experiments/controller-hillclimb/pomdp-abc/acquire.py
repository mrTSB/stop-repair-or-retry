"""Isolated paired trees for a POMDP planner, an LLM judge, and their hybrid."""

import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path[:0] = [str(PARENT / 'retry-gate/paired-uplift/margin-gate'), str(PARENT / 'retry-gate/paired-uplift')]
from study_data import DATA, LAST, MODEL, MODEL_DIGEST, ROOT, existing_questions, gsm8k, prior_pairs, prior_test, question_rows, records, science, stem
from compact import parse_answer, prompt as compact_prompt
from collect import SCIENCE_SUFFIX, request
from probe import confidence

JUDGE = 'qwen3.5:9b'
JUDGE_DIGEST = '6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7'
SEED = 2026111905
PATHS = ('', 'repair', 'retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry')
CHECKPOINTS = PATHS[:3]
PREVIOUS = ROOT / 'results/controller-hillclimb/retry-gate/paired-uplift/margin-gate/data'
SOLVER_CAP = 128
JUDGE_CAP = 128


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            return [json.loads(line) for line in stream if line.strip()]
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paired_archive():
    rows = prior_pairs()
    extra = {row['id']: row['short'] for row in records(PREVIOUS, 'prior-retries')}
    for row in prior_test():
        if row['short'] is None:
            row['short'] = extra[row['id']]
        rows.append(row)
    return rows


def balanced(rows, size, label, counts=None):
    picked = []
    if counts is None:
        counts = {'math': size // 2, 'biology': size // 6, 'chemistry': size // 6, 'physics': size // 6}
    if sum(counts.values()) != size:
        raise ValueError('Subject counts must cover the requested size')
    for subject, count in counts.items():
        group = sorted((row for row in rows if row['subject'] == subject),
                       key=lambda row: hashlib.sha256(f'{SEED}:{label}:{row["id"]}'.encode()).digest())
        if len(group) < count:
            raise ValueError(f'Insufficient unused {subject} questions. Available subjects: {dict(Counter(r["subject"] for r in rows))}')
        picked.extend(group[:count])
    random.Random(f'{SEED}:{label}').shuffle(picked)
    return picked


def prepare(args):
    archive = paired_archive()
    dev_questions = question_rows(PREVIOUS, 'dev')
    dev_signals = {r['id']: r for r in records(PREVIOUS, 'dev-pairs')}
    dev = balanced([{**row, **dev_signals[row['id']]} for row in dev_questions], 120, 'dev')
    train = balanced(archive, 144, 'train')
    history = existing_questions() + archive
    history += [row for base in (LAST, PREVIOUS) for split in ('train', 'dev', 'test')
                if (base / 'splits' / f'{split}-questions.jsonl').exists() for row in question_rows(base, split)]
    ids, texts = {r['id'] for r in history}, {stem(r['question']) for r in history}
    exclusions = json.loads((DATA.parent / 'provenance/selection-manifest.json').read_bytes())
    ids.update(f'gsm8k-train-{i:05d}' for i in exclusions['historical_math_training_exclusions'])
    ids.update(exclusions['historical_math_test_exclusions'])
    ids.update(exclusions['historical_science_exclusions'])
    ids.update(item['id'] for item in exclusions['source_quality_exclusions'])
    pool = gsm8k(args.math_train, 'train')
    pool += [row for group in science(args.science).values() for row in group]
    available = []
    for row in sorted(pool, key=lambda r: hashlib.sha256(f'{SEED}:pool:{r["id"]}'.encode()).digest()):
        text = stem(row['question'])
        if row['id'] not in ids and text not in texts:
            available.append(row)
            ids.add(row['id'])
            texts.add(text)
    test = balanced(available, 420, 'test', {'math': 210, 'biology': 28, 'chemistry': 91, 'physics': 91})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'splits').mkdir()
    manifest = {'seed': SEED, 'math_source': 'Unused official GSM8K training questions form the study holdout.',
                'science_source': 'Unused MMLU-Pro public test questions.', 'dev_source': '120 prior development questions.',
                'source_sha256': {'math_train': digest(args.math_train), 'science': digest(args.science)}, 'splits': {}}
    for split, rows in [('train', train), ('dev', dev), ('test', test)]:
        questions = [{key: row[key] for key in ('id', 'domain', 'subject', 'question', 'answer')} for row in rows]
        path = args.output / 'splits' / f'{split}-questions.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in questions))
        manifest['splits'][split] = {'n': len(rows), 'subjects': dict(Counter(r['subject'] for r in rows)),
                                    'sha256': digest(path)}
        if split != 'test':
            (args.output / f'{split}-initial.jsonl').write_text(''.join(json.dumps({'id': row['id'], 'compact': row['compact']}) + '\n' for row in rows))
    (args.output / 'splits/selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


def chat(text):
    return f'<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def solver_prompt(row, path, tree):
    if not path or path.split('/')[-1] == 'retry':
        return compact_prompt(row['domain'], row['question'])
    parent = path.rsplit('/', 1)[0] if '/' in path else ''
    unit = 'one option letter' if row['domain'] == 'science' else 'a number'
    task = row['question'].removesuffix(SCIENCE_SUFFIX)
    return chat(f'{task}\n\nPrevious solution:\n{tree[parent]["text"]}\n\n'
                'Check the previous solution for factual, reasoning, and arithmetic errors. '
                'Recompute any uncertain step. Keep a correct answer, and correct any error you find. '
                f'Reply with one short line of scratch work and then "Final: X", where X is {unit}. Keep the reply under 60 words.')


def judge_prompt(row, path, tree):
    parts = path.split('/') if path else []
    history = ['']
    if parts:
        history.extend('/'.join(parts[:i]) for i in range(1, len(parts) + 1))
    seen = '\n'.join(f'Attempt {i+1} ({key or "initial"}):\n{tree[key]["text"]}' for i, key in enumerate(history))
    task = row['question'].removesuffix(SCIENCE_SUFFIX)
    return chat(
        'You control a fixed solver. Assess whether its CURRENT answer is correct and select the next action. '
        'STOP submits the current answer. REPAIR asks the solver to check its solution. RETRY starts a fresh solution. '
        f'There are {2-len(parts)} solver calls remaining. '
        'Maximize the chance of a correct submitted answer minus 0.05 per 1000 parameter-weighted tokens. '
        'A retry typically costs 250 tokens and a repair 350. Your call is charged too. '
        'Use STOP when the answer is correct, REPAIR for a localized fix, and RETRY for a mistaken approach. '
        'Return ONLY a compact JSON object. First include "check", a verification in at most 12 words. '
        'Then include "p_correct" between 0 and 1 and "action" equal to "stop", "repair", or "retry". '
        'Example format: {"check":"The equation and answer are consistent.","p_correct":0.95,"action":"stop"}.\n\n'
        f'{task}\n\n{seen}')


def call(row, path, tree, judge=False):
    model, fingerprint = (JUDGE, JUDGE_DIGEST) if judge else (MODEL, MODEL_DIGEST)
    formatted = judge_prompt(row, path, tree) if judge else solver_prompt(row, path, tree)
    seed = int(hashlib.sha256(f'{SEED}:{row["id"]}:{path}'.encode()).hexdigest()[:8], 16) % (2**31)
    started = time.monotonic()
    response = request('/api/generate', {'model': model, 'prompt': formatted, 'raw': True, 'stream': False,
                       'logprobs': not judge, 'keep_alive': '30m',
                       'options': {'temperature': 0 if judge or not path else .65, 'seed': seed,
                                   'num_predict': JUDGE_CAP if judge else SOLVER_CAP, 'num_ctx': 4096}})
    if not response.get('done') or response.get('model') != model:
        raise RuntimeError('Incomplete local model response')
    text = response['response']
    result = {'model': model, 'model_digest': fingerprint,
              'prompt_sha256': hashlib.sha256(formatted.encode()).hexdigest(),
              'text': text, 'finish': response['done_reason'], 'input_tokens': int(response['prompt_eval_count']),
              'output_tokens': int(response.get('eval_count', 0)), 'latency_seconds': time.monotonic()-started,
              'seed': seed, 'temperature': 0 if judge or not path else .65}
    if judge:
        obj = {}
        decoder = json.JSONDecoder()
        for match in re.finditer(r'\{', text):
            try:
                candidate, _ = decoder.raw_decode(text[match.start():])
                if isinstance(candidate, dict):
                    obj = candidate
            except json.JSONDecodeError:
                continue
        action = str(obj.get('action', '')).lower()
        p = obj.get('p_correct')
        result.update(action=action if action in ('stop', 'repair', 'retry') else 'stop',
                      p_correct=float(p) if type(p) in (int, float) and 0 <= p <= 1 else .5,
                      parse_valid=action in ('stop', 'repair', 'retry') and type(p) in (int, float) and 0 <= p <= 1)
    else:
        result.update(answer=parse_answer(row['domain'], text),
                      confidence=confidence(text, response.get('logprobs', []), row['domain']))
    return result


def initial_rows(source, split):
    questions = read(source / 'splits' / f'{split}-questions.jsonl')
    initial = source / f'{split}-initial.jsonl'
    signals = {r['id']: r['compact'] for r in read(initial)} if initial.exists() else {}
    return [{**q, 'initial': signals.get(q['id'])} for q in questions]


def collect(args):
    source = args.data
    rows = initial_rows(source, args.split)
    tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    for model, expected in ((MODEL, MODEL_DIGEST), (JUDGE, JUDGE_DIGEST)):
        if tags.get(model) != expected:
            raise ValueError(f'Installed weights changed for {model}')
    target = source / f'{args.split}-{args.action}.jsonl'
    if args.action in ('judges','probes'):
        solver = {r['id']: r['tree'] for r in read(source / f'{args.split}-trees.jsonl')}
    if args.action == 'probes':
        rows = [row for row in rows if 'confidence' not in solver[row['id']]['']]
    with target.open('a+', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        prior = [json.loads(line) for line in stream]
        saved = {(r['id'], r.get('path', '')):r for r in prior}
        expected = {r['id'] for r in rows}
        if len(saved) != len(prior) or any(k[0] not in expected for k in saved):
            raise ValueError('Unexpected duplicate or foreign collection rows')
        for i, row in enumerate(rows, 1):
            if args.action == 'trees':
                if (row['id'], '') in saved:
                    continue
                tree = {'': row['initial']} if row['initial'] else {}
                for path in PATHS:
                    if path not in tree:
                        tree[path] = call(row, path, tree)
                stream.write(json.dumps({'id': row['id'], 'tree': tree}) + '\n')
                stream.flush()
            elif args.action == 'judges':
                for path in CHECKPOINTS:
                    if (row['id'], path) not in saved:
                        verdict = call(row, path, solver[row['id']], judge=True)
                        stream.write(json.dumps({'id': row['id'], 'path': path, 'judge': verdict}) + '\n')
                        stream.flush()
            elif (row['id'],'') not in saved:
                first = call(row,'',{})
                before = solver[row['id']]['']
                checked = ('text','answer','finish','input_tokens','output_tokens','prompt_sha256')
                matches = all(first[key]==before[key] for key in checked)
                stream.write(json.dumps({'id':row['id'],'compact':first,'matches_archive':matches})+'\n')
                stream.flush()
            if i % 12 == 0 or i == len(rows):
                print(f'{args.split} {args.action} {i}/{len(rows)}', flush=True)


def collect_holdout(args):
    from belief import BeliefPlanner, evaluate
    from study import evidence

    tags = {row['name']:row['digest'] for row in request('/api/tags')['models']}
    if tags.get(JUDGE)!=JUDGE_DIGEST or tags.get(MODEL)!=MODEL_DIGEST:
        raise ValueError('Installed model weights changed after policy freeze')
    frozen_path = args.frozen
    frozen = json.loads(frozen_path.read_bytes())
    if frozen['evidence'] != evidence(args.data):
        raise ValueError('Training or policy code differs from the frozen experiment')
    models = {key:BeliefPlanner.restore(value) for key,value in frozen['models'].items()}
    methods = [(name, models[item['fit']], item['settings']) for name,item in frozen['selected'].items()]
    control = frozen['random_judge_control']
    c = frozen['selected']['C']
    methods += [('random',models[control['fit']],control['settings']),
                ('never',models[c['fit']],c['settings']),('always',models[c['fit']],c['settings'])]
    rows = read(args.data/'splits/test-questions.jsonl')
    metadata = args.data/'test-collection.json'
    expected = {'frozen_sha256':digest(frozen_path)}
    if metadata.exists() and json.loads(metadata.read_bytes())!=expected:
        raise ValueError('The frozen test collection policy changed')
    metadata.write_text(json.dumps(expected,indent=2)+'\n')
    with ExitStack() as stack:
        streams = {}
        saved = {}
        for name in ('test-judges','test-trees','test-solver-observations'):
            stream = stack.enter_context((args.data/f'{name}.jsonl').open('a+',encoding='utf-8'))
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            stream.seek(0)
            saved[name] = [json.loads(line) for line in stream]
            streams[name] = stream
        stored = saved['test-judges']
        judges = {(r['id'],r['path']):r['judge'] for r in stored}
        if len(judges)!=len(stored):
            raise ValueError('Duplicate test judge observations')
        trees = {r['id']:r['tree'] for r in saved['test-trees']}
        completed = set(trees)
        journal_keys = set()
        for record in saved['test-solver-observations']:
            key = record['id'],record['path']
            if key in journal_keys:
                raise ValueError('Duplicate solver observations in the recovery journal')
            journal_keys.add(key)
            tree = trees.setdefault(record['id'],{})
            if record['path'] in tree and tree[record['path']]!=record['node']:
                raise ValueError('A saved solver response differs from its recovery journal')
            tree[record['path']] = record['node']
        allowed = {r['id'] for r in rows}
        if not set(trees)<=allowed or any(key[0] not in allowed or key[1] not in CHECKPOINTS for key in judges):
            raise ValueError('Unexpected question IDs in held-out observations')

        for i,row in enumerate(rows,1):
            visible = {**row,'answer':''}
            tree = trees.setdefault(row['id'],{})

            def store_solver(path):
                node = call(visible,path,tree)
                streams['test-solver-observations'].write(json.dumps({'id':row['id'],'path':path,'node':node})+'\n')
                streams['test-solver-observations'].flush()
                tree[path] = node

            if '' not in tree:
                store_solver('')
            for config in frozen['baselines'].values():
                if config['action']=='stop':
                    continue
                for depth in range(1,config['depth']+1):
                    path = '/'.join([config['action']]*depth)
                    if path not in tree:
                        store_solver(path)
            for family,model,settings in methods:
                while True:
                    try:
                        evaluate(visible,tree,judges,model,settings,family)
                        break
                    except KeyError as missing:
                        key = missing.args[0]
                        if isinstance(key,str) and key in PATHS:
                            store_solver(key)
                            continue
                        if not isinstance(key,tuple) or key[0]!=row['id'] or key[1] not in CHECKPOINTS:
                            raise
                        verdict = call(visible,key[1],tree,judge=True)
                        judges[key] = verdict
                        streams['test-judges'].write(json.dumps({'id':key[0],'path':key[1],'judge':verdict})+'\n')
                        streams['test-judges'].flush()
            if row['id'] not in completed:
                streams['test-trees'].write(json.dumps({'id':row['id'],'tree':tree})+'\n')
                streams['test-trees'].flush()
                completed.add(row['id'])
            if i%12==0 or i==len(rows):
                solver_count = sum(len(value) for value in trees.values())
                print(f'holdout {i}/{len(rows)} questions, {solver_count} solver responses, {len(judges)} judge responses',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prepare_args = sub.add_parser('prepare')
    prepare_args.add_argument('--math-train', type=Path, required=True)
    prepare_args.add_argument('--science', type=Path, required=True)
    prepare_args.add_argument('--output', type=Path, required=True)
    for action in ('trees', 'judges', 'probes'):
        opts = sub.add_parser(action)
        opts.add_argument('--data', type=Path, required=True)
        opts.add_argument('--split', choices=('train', 'dev', 'test'), required=True)
    selected = sub.add_parser('holdout')
    selected.add_argument('--data',type=Path,required=True)
    selected.add_argument('--frozen',type=Path,required=True)
    args = parser.parse_args()
    if args.action=='prepare':
        prepare(args)
    elif args.action=='holdout':
        collect_holdout(args)
    else:
        collect(args)


if __name__ == '__main__':
    main()
