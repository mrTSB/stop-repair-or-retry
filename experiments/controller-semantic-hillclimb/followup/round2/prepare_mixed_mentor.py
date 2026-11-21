"""Fit the new retained-review hybrid with locally generated, excluded-question observations.

All inputs and text-bearing caches stay local. New questions use the old-only
full observer; the 64 legacy questions use their original excluded folds.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import ordinary, committed, canonical_review
from difficulty_pomdp import FIXED, PLAIN
from semantic_data import read
from scoring import correct
from reasoning_pilot import short_repair
from network_large import encode, predict
import dynamic_grounded_pomdp as perception
import mentor_pomdp
import extended_acquire as old
import mentor_pairs
import hybrid_repairability


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def paid_prefix(tree, path):
    """The observer receives only the current response and its paid ancestors."""
    paths = ['']
    if path:
        parts = path.split('/')
        paths.extend('/'.join(parts[:i]) for i in range(1, len(parts) + 1))
    return {p: tree[p] for p in paths}


def plain_data(data, split):
    """Read the sealed plain prefix while an ordinary collector may append later rows."""
    receipt = json.loads((data / (split + '-plain-complete.json')).read_bytes())
    raw = (data / (split + '-observations.jsonl')).read_bytes()
    prefix = b''.join(raw.splitlines(keepends=True)[:receipt['base_records']])
    if hashlib.sha256(prefix).hexdigest() != receipt['base_bank_sha256']:
        raise ValueError('The completed plain response prefix changed')
    rows = read(data / 'splits' / (split + '-questions.jsonl'))
    trees = {r['id']: {} for r in rows}
    for line in prefix.splitlines():
        item = json.loads(line)
        if item['kind'] == 'solver' and item['path'] in PLAIN:
            tree = trees[item['id']]
            if item['path'] in tree:
                raise ValueError('Duplicate sealed plain solver response')
            tree[item['path']] = item['response']
    if any(set(tree) != set(PLAIN) for tree in trees.values()):
        raise ValueError('The completed plain bank lacks a registered branch')
    return rows, trees


def overlay(rows, source_trees, records, selected, validate=True):
    """Remove ordinary guided branches before overlaying the strong-review cohort."""
    by_id = {r['id']: r for r in rows}
    selected = {r['id'] for r in selected}
    trees = {r['id']: {p: n for p, n in source_trees[r['id']].items() if p in PLAIN} for r in rows}
    judges, seen = {}, set()
    for item in records:
        qid, kind, path, node = item['id'], item['kind'], item['path'], item['response']
        key = qid, kind, path
        if item['dose'] != 1024 or qid not in selected or key in seen:
            raise ValueError('Unexpected or duplicate mixed-mentor receipt')
        seen.add(key)
        if kind == 'judge' and path == '':
            judges[qid, ''] = canonical_review(node)
        elif kind == 'solver' and path in ('guided', 'guided/reuse', 'guided/retry', 'retry/reuse'):
            trees[qid][path] = node
        else:
            raise ValueError('Unexpected mixed-mentor branch')
    if len(seen) != 5 * len(selected) or len(judges) != len(selected):
        raise ValueError('Finish every registered mixed-mentor pair before fitting')
    for qid in selected:
        tree, row, judge = trees[qid], {**by_id[qid], 'answer': ''}, judges[qid, '']
        if not all(p in tree for p in FIXED):
            raise ValueError('Missing fixed training control')
        if not validate:
            continue
        mentor_pairs.verify_review(row, tree[''], judge, 1024, 512)
        for path in ('guided', 'guided/reuse', 'retry/reuse', 'guided/retry'):
            node = tree[path]
            if path == 'guided/retry':
                prompt, cap = old.retry_prompt(row), old.SOLVER_CAP
            else:
                parent = '' if path == 'guided' else path.rsplit('/', 1)[0]
                prompt, cap = short_repair(row, tree[parent], judge), 128
            expected = hashlib.sha256(prompt.encode()).hexdigest()
            # Archived ordinary solver receipts predate the output_cap field.
            # Only a pure retry may inherit that fixed256cap contract.
            actual_cap = node.get('output_cap', cap if path == 'guided/retry' else None)
            if (node['prompt_sha256'], node['seed'], node['model_digest'], actual_cap) != (
                    expected, mentor_pairs.seed_for(qid, path), old.MODEL_DIGEST, cap):
                raise ValueError('Mixed-mentor response differs from its paid prompt or seed')
            if not 0 <= node['output_tokens'] <= cap:
                raise ValueError('Mixed-mentor solver exceeded its output contract')
    return trees, judges


def training_view(args):
    completion = json.loads((args.mentor / 'complete.json').read_bytes())
    contract = json.loads((args.mentor / 'contract.json').read_bytes())
    if completion['bank_sha256'] != sha(args.mentor / 'train.jsonl') or completion['contract_sha256'] != sha(args.mentor / 'contract.json'):
        raise ValueError('Mixed-mentor bank changed after completion')
    legacy, old_trees, _, _ = ordinary(args.old_data, 'train')
    new, new_trees = plain_data(args.new_data, 'train')
    selected = contract['selection']
    old_ids = {r['id'] for r in selected if r['bank'] == 'old'}
    if len(old_ids) != 64 or len(new) != 512 or len(selected) != 192:
        raise ValueError('Use the registered 64 old +128 new mentor cohort and all512new training questions')
    if {r['id'] for r in legacy} & {r['id'] for r in new}:
        raise ValueError('Old and new training IDs overlap')
    rows = new + [r for r in legacy if r['id'] in old_ids]
    trees, judges = overlay(rows, {**old_trees, **new_trees}, committed(args.mentor / 'train.jsonl'), selected)
    return rows, trees, judges, old_ids, contract


def entries(rows, trees, judges, old_ids, observer):
    metadata, inputs = [], []
    for row in rows:
        qid, tree = row['id'], trees[row['id']]
        hard = int(np.mean([correct(tree[p], row['answer']) for p in FIXED]) < .6)
        fold = int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) % 3 if qid in old_ids else 'full'
        nodes = [('solver', p, n) for p, n in tree.items()]
        if (qid, '') in judges:
            nodes.append(('original-review', '', judges[qid, '']))
        for kind, path, node in nodes:
            safe, paid = {**row, 'answer': ''}, paid_prefix(tree, path)
            entry = observer.entry(safe, path, paid, None if kind == 'solver' else node)
            inputs.append(entry)
            metadata.append({'id': qid, 'kind': kind, 'path': path, 'fold': fold,
                             'train': kind == 'solver' and path in ('', 'repair', 'retry', 'guided'),
                             'valid': bool(node['answer']), 'label': 2 * hard + int(correct(node, row['answer'])),
                             'prompt_sha256': node['prompt_sha256'], 'input_sha256': fingerprint(entry)})
    return metadata, inputs


def neural(args, rows, trees, judges, old_ids):
    recipe = json.loads((args.verifier / 'full/recipe.json').read_bytes())
    legacy = read(args.legacy_examples)
    legacy_ids = {r['id'] for r in legacy}
    if recipe['data_sha256'] != sha(args.legacy_examples) or not old_ids <= legacy_ids:
        raise ValueError('The old-only observer provenance does not match its declared training examples')
    if (set(r['id'] for r in rows) - old_ids) & legacy_ids:
        raise ValueError('A supposedly unseen new question was used in the full observer')
    temperature = json.loads((args.verifier / 'training-summary.json').read_bytes())['temperature_selected_on_training_oof']
    observer = perception.Observer(str((args.verifier / 'full').resolve()), temperature, device=args.device)
    metadata, inputs = entries(rows, trees, judges, old_ids, observer)
    outputs = {k: np.full((len(inputs), width), np.nan, dtype=np.float32)
               for k, width in (('state', 4), ('plain', 8), ('guided', 8))}
    done = np.zeros(len(inputs), dtype=bool)
    # The previous A8 fit evaluated these exact new nodes with this full model.
    reused = 0
    base_identity = json.loads((args.base / 'identity.json').read_bytes())
    if base_identity['encoder_recipe_sha256'] != sha(args.verifier / 'full/recipe.json'):
        raise ValueError('A8 and mentor must use identical observer assets')
    base_meta = read(args.base / 'training-neural.jsonl')
    base_outputs = np.load(args.base / 'training-neural.npz')
    lookup = {(r['id'], r['path'], r['prompt_sha256'], r['input_sha256']): i for i, r in enumerate(base_meta)}
    for i, r in enumerate(metadata):
        key = r['id'], r['path'], r['prompt_sha256'], r['input_sha256']
        if r['fold'] == 'full' and r['kind'] == 'solver' and key in lookup:
            for name in outputs:
                outputs[name][i] = base_outputs[name][lookup[key]]
            done[i] = True
            reused += 1
    model_receipts = {}
    for fold in ('full', 0, 1, 2):
        directory = args.verifier / ('full' if fold == 'full' else f'folds/{fold}')
        current_recipe = json.loads((directory / 'recipe.json').read_bytes())
        if current_recipe['fold'] != fold or current_recipe['data_sha256'] != sha(args.legacy_examples):
            raise ValueError('Incorrect excluded-question checkpoint')
        model_receipts[str(fold)] = sha(directory / 'recipe.json')
        indexes = [i for i, r in enumerate(metadata) if r['fold'] == fold and not done[i]]
        if not indexes:
            continue
        chunk_path = args.output / ('neural-' + str(fold) + '.npz')
        chunk_receipt = args.output / ('neural-' + str(fold) + '.json')
        identity = {'recipe_sha256': model_receipts[str(fold)],
                    'inputs_sha256': fingerprint([metadata[i] for i in indexes]), 'n': len(indexes)}
        if chunk_receipt.exists():
            saved = json.loads(chunk_receipt.read_bytes())
            if saved['identity'] != identity or saved['outputs_sha256'] != sha(chunk_path):
                raise ValueError('Changed local neural chunk')
            chunk = np.load(chunk_path)
        else:
            observer.directory = str(directory.resolve())
            observer.load()
            tensors = encode(observer._tokenizer, [inputs[i] for i in indexes], current_recipe['max_length'])
            values = predict(observer._model, tensors, args.device, batch=8)
            chunk = dict(zip(('state', 'plain', 'guided'), values))
            np.savez_compressed(chunk_path, **chunk)
            chunk_receipt.write_text(json.dumps({'identity': identity, 'outputs_sha256': sha(chunk_path)}, indent=2) + '\n')
            observer._model = None
            observer._tokenizer = None
            observer._recipe = {}
            del tensors, values
            gc.collect()
        for name in outputs:
            outputs[name][indexes] = chunk[name]
        done[indexes] = True
        print(f'Prepared excluded-question mentor observations for {fold}: {len(indexes)}', flush=True)
    if not done.all() or any(not np.isfinite(v).all() for v in outputs.values()):
        raise ValueError('Incomplete local neural observations')
    np.savez_compressed(args.output / 'neural.npz', **outputs)
    (args.output / 'metadata.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in metadata))
    receipt = {'n': len(metadata), 'reused_full_observer_nodes': reused, 'encoder_recipes': model_receipts,
               'inputs_sha256': fingerprint(inputs), 'metadata_sha256': sha(args.output / 'metadata.jsonl'),
               'outputs_sha256': sha(args.output / 'neural.npz'), 'reference_answers_in_inputs': False,
               'text_uploaded': False, 'legacy_questions': len(old_ids), 'unseen_new_questions': len(rows) - len(old_ids)}
    (args.output / 'neural-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return metadata, outputs


def run(args):
    while not (args.mentor / 'complete.json').exists():
        if not args.wait:
            raise ValueError('Mixed mentor collection is not yet complete; use --wait')
        time.sleep(30)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, trees, judges, old_ids, contract = training_view(args)
    selected_a = json.loads((args.base / 'summary.json').read_bytes())['selected_A']
    base_path = args.base / (selected_a['model'] + '.pkl')
    identity = {'source_sha256': sha(__file__), 'mentor_contract_sha256': sha(args.mentor / 'contract.json'),
                'mentor_bank_sha256': sha(args.mentor / 'train.jsonl'), 'base_model_sha256': sha(base_path),
                'legacy_examples_sha256': sha(args.legacy_examples),
                'new_train_split_sha256': sha(args.new_data / 'splits/train-questions.jsonl'),
                'tree_sha256': fingerprint(trees), 'model_files': {str(p): sha(p) for p in
                   (Path(mentor_pomdp.__file__), Path(hybrid_repairability.__file__))},
                'fit_questions': len(rows), 'mentor_questions': len(contract['selection']),
                'development_or_test_used': False}
    identity_path = args.output / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_bytes()) != identity:
        raise ValueError('Changed mentor fitting inputs; use a new output directory')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    metadata, outputs = neural(args, rows, trees, judges, old_ids)
    models = []
    for smoothing in args.smoothing:
        for semantic in (False, True):
            key = f'smooth{smoothing:g}-review{int(semantic)}'
            mentor = mentor_pomdp.fit(rows, trees, judges, args.verifier / 'full', metadata, outputs,
                                     smoothing=smoothing, semantic_review=semantic)
            mentor.fit_receipt.update(mixed_new_legacy_training=True, new_questions=512, legacy_questions=64,
                                     neural_provenance=identity, excluded_legacy_questions=True,
                                     full_observer_never_trained_on_new_questions=True)
            base = pickle.loads(base_path.read_bytes())
            hybrid = hybrid_repairability.build(base, mentor, rows, trees, judges, smoothing=smoothing)
            path = args.output / (key + '.pkl')
            path.write_bytes(pickle.dumps(hybrid, protocol=5))
            models.append({'key': key, 'sha256': sha(path), 'smoothing': smoothing, 'semantic_review': semantic})
            print('Fitted mixed retained-mentor hybrid', key, flush=True)
    summary = {'fit_questions': len(rows), 'paired_review_questions': len(judges), 'models': models,
               'development_or_test_used': False, 'base_A': selected_a, 'identity_sha256': sha(identity_path)}
    (args.output / 'complete.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('old-data', 'new-data', 'mentor', 'verifier', 'legacy-examples', 'base', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    p.add_argument('--smoothing', type=float, nargs='+', default=(8., 32.))
    p.add_argument('--wait', action='store_true')
    run(p.parse_args())
