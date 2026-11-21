"""Refit selected POMDPs from training-only OOF predictions before opening the holdout."""

import argparse
import importlib
import json
from pathlib import Path
import pickle
import re

import numpy as np

from semantic_data import read, records
from late_pairs import merged_records
from semantic_study_common import evidence, models, run_policy, sha, statistics


def run(args):
    frozen = json.loads((args.output / 'frozen.json').read_bytes())
    if frozen['evidence'] != evidence(args.data):
        raise ValueError('Changed frozen evidence')
    train, tt, tj = merged_records(args.data, 'train', False)
    dev, dt, original = records(args.data, 'dev', False)
    _, dt, late = merged_records(args.data, 'dev', False)
    planners = models(frozen, args.output)
    refitted = {}
    for family, model in planners.items():
        config = frozen['selected'][family]
        module = importlib.import_module(config['module'])
        source = Path(config['probe']) / (config['model'] + '.pkl')
        original_model = pickle.loads(source.read_bytes())
        verifier = Path(original_model.classifier.directory).parent
        contract = json.loads((verifier / 'contract.json').read_bytes())
        examples = args.grounded_examples if contract.get('input_profile') else args.examples
        if sha(examples / 'train.jsonl') != contract['data_sha256']:
            raise ValueError('Changed training examples')
        metadata = read(examples / 'train.jsonl')
        assert {r['id'] for r in metadata} <= {r['id'] for r in train}
        smoothing = float(re.match(r'smooth([\d.]+)-', config['model'])[1])
        fresh = module.fit(train, tt, tj, verifier / 'full', metadata, np.load(verifier / 'oof.npz'),
                           smoothing=smoothing, late=config['protocol'] == 'late',
                           semantic_review=model.review_observer is not None)
        model.classifier.warm(dev, dt, late)
        model.classifier.warm(dev, dt, original)
        for attr in ('_model', '_tokenizer', '_recipe', '_cache', 'device', '_retriever', '_backgrounds'):
            if hasattr(model.classifier, attr):
                setattr(fresh.classifier, attr, getattr(model.classifier, attr))
        fresh.review_observer = fresh.classifier if model.review_observer is not None else None
        refitted[family] = fresh
    comparisons = blinded = 0
    for family in ('A', 'B', 'C'):
        config = frozen['selected'][family]
        items = []
        for row in dev:
            result = run_policy(family, row, dt[row['id']], original, late, planners.get(family), config)
            items.append(result)
            if family in refitted:
                independent = run_policy(family, row, dt[row['id']], original, late, refitted[family], config)
                assert result == independent, (family, row['id'], 'refit')
                comparisons += 1
            parts = result['path'].split('/') if result['path'] else []
            visited = [''] + ['/'.join(parts[:i]) for i in range(1, len(parts) + 1)]
            acquired = [d['path'] for d in result['decisions'] if d['queried']]
            paid_tree = {p: dt[row['id']][p] for p in visited}
            active = late if config['protocol'] == 'late' else original
            paid_reviews = {(row['id'], p): active[row['id'], p] for p in acquired}
            expected = {k: v for k, v in result.items() if k != 'correct'}
            for reference in ('', '918273645546372819'):
                model = refitted.get(family)
                if model is not None:
                    for attr in ('_observed', '_current_context', '_review_context'):
                        getattr(model, attr, {}).clear()
                    getattr(model.classifier, '_probability_cache', {}).clear()
                actual = run_policy(family, {**row, 'answer': reference}, paid_tree,
                                    paid_reviews, paid_reviews, model, config)
                actual.pop('correct')
                assert actual == expected, (family, row['id'], 'blind')
                blinded += 1
        measured = statistics(items)
        expected = config['dev']
        assert set(measured) == set(expected)
        assert all(np.isclose(measured[k], expected[k], rtol=1e-7, atol=1e-7) for k in measured), family
        print(f'Reproduced {family}: {measured["correct"]}/{len(dev)}', flush=True)
    receipt = {'passed': True, 'freeze_sha256': sha(args.output / 'frozen.json'),
               'training_only_refitted_outcomes': comparisons, 'blind_pruned_replays': blinded,
               'fresh_data_used': False}
    (args.output / 'development-reproduction.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--examples', type=Path, required=True)
    parser.add_argument('--grounded-examples', type=Path, required=True)
    run(parser.parse_args())
