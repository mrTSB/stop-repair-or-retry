"""Reconcile development outcomes and replay policies using only paid, blinded inputs."""

import argparse
import importlib
import json
from pathlib import Path
import pickle

import numpy as np

from semantic_data import read, records, write
from late_pairs import merged_records, llm_policy
from context_pomdp import evaluate_llm
from study_common import BASELINES, statistics, sha
from study import fixed
from scoring import correct


def run(args):
    semantic = importlib.import_module(args.module)
    summary = json.loads((args.output / 'summary.json').read_bytes())
    rows, trees, original = records(args.data, 'dev', False)
    _, trees, late = merged_records(args.data, 'dev', False)
    models = {}
    shared = None
    all_items = []
    charge_checks = blinded = 0
    for family in (*BASELINES, 'A', 'B', 'C', 'C_cheaper_than_B'):
        setting = summary['selected'].get(family)
        if family not in BASELINES and setting is None:
            continue
        model = None
        if family in ('A', 'C', 'C_cheaper_than_B'):
            key = setting['model']
            if key not in models:
                model = pickle.loads((args.output / (key + '.pkl')).read_bytes())
                model.classifier.device = args.device
                if shared is None:
                    model.classifier.warm(rows, trees, late)
                    model.classifier.warm(rows, trees, original)
                    shared = model.classifier
                else:
                    for attribute in ('_model', '_tokenizer', '_recipe', '_cache', 'device',
                                      '_retriever', '_backgrounds'):
                        if hasattr(shared, attribute):
                            setattr(model.classifier, attribute, getattr(shared, attribute))
                models[key] = model
            model = models[key]
        reviews = late if setting and setting['late'] else original

        def policy(row, tree, paid_reviews):
            if family in BASELINES:
                return fixed(row, tree, *BASELINES[family])
            if family == 'B':
                return (llm_policy if setting['late'] else evaluate_llm)(row, tree, paid_reviews, setting['settings'])
            return semantic.evaluate(row, tree, paid_reviews, model, setting['settings'],
                                     'A' if family == 'A' else 'C')

        def clear():
            if model is not None:
                for attribute in ('_observed', '_current_context', '_review_context'):
                    getattr(model, attribute, {}).clear()
                getattr(model.classifier, '_probability_cache', {}).clear()

        items = []
        for row in rows:
            clear()
            item = policy(row, trees[row['id']], reviews)
            items.append(item)
            parts = item['path'].split('/') if item['path'] else []
            visited = [''] + ['/'.join(parts[:i]) for i in range(1, len(parts) + 1)]
            assert len(visited) <= 3
            decisions = item.get('decisions', [])
            acquired = [d['path'] for d in decisions if d['queried']]
            assert len(acquired) == len(set(acquired)) <= 2
            assert set(acquired) <= set(visited)
            if family == 'A':
                assert not acquired
            paid_tree = {p: trees[row['id']][p] for p in visited}
            paid_reviews = {(row['id'], p): reviews[row['id'], p] for p in acquired}
            solver = sum(n['input_tokens'] + n['output_tokens'] for n in paid_tree.values())
            judge = sum(n['input_tokens'] + n['output_tokens'] for n in paid_reviews.values())
            assert (item['solver_tokens'], item['judge_tokens'], item['solver_calls'], item['judge_calls']) == (
                solver, judge, len(visited), len(acquired))
            verifier = 0.
            if model is not None:
                calls = len(decisions) + (len(acquired) if model.review_observer is not None else 0)
                slots = (item['verifier_tokens'] if model.fit_receipt.get('dynamic_padding')
                         else calls * model.classifier._recipe['max_length'])
                assert item['verifier_calls'] == calls
                assert item['verifier_token_slots'] == slots
                verifier = model.classifier._recipe['parameters'] / 1e9 * slots
            assert np.isclose(item['compute'], (4 * solver + 3 * judge + verifier) / 3.09)
            assert np.isclose(item.get('total_parameter_compute', item['compute']),
                              (4 * solver + 35 * judge + verifier) / 3.09)
            assert item['correct'] == int(correct(paid_tree[item['path']], row['answer']))
            charge_checks += 1
            if family not in BASELINES:
                expected = {k: v for k, v in item.items() if k != 'correct'}
                for reference in ('', '918273645546372819'):
                    clear()
                    actual = policy({**row, 'answer': reference}, paid_tree, paid_reviews)
                    actual.pop('correct')
                    assert actual == expected, (family, row['id'])
                    blinded += 1
            all_items.append({'policy': family, **item})
        measured = statistics(items)
        expected = summary['baselines'][family] if family in BASELINES else setting['dev']
        assert set(measured) == set(expected)
        assert all(np.isclose(measured[k], expected[k], rtol=1e-7, atol=1e-7) for k in measured), (family, measured, expected)
        print(f'Audited {family}: {measured["correct"]}/{len(rows)}', flush=True)
    write(args.output / 'audited-per-question.jsonl', all_items)
    receipt = {'passed': True, 'split': 'development; exposed panels only', 'fresh_data_used': False,
               'questions': len(rows), 'independent_charge_checks': charge_checks,
               'blinded_pruned_policy_replays': blinded,
               'source_sha256': sha(Path(semantic.__file__)),
               'summary_sha256': sha(args.output / 'summary.json'),
               'outcomes_sha256': sha(args.output / 'audited-per-question.jsonl')}
    (args.output / 'audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--module', choices=('semantic_pomdp', 'stable_pomdp', 'grounded_pomdp', 'tree_context_pomdp', 'calibrated_review_pomdp', 'stacked_pomdp', 'efficient_stacked_pomdp', 'dynamic_grounded_pomdp'), default='semantic_pomdp')
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    run(parser.parse_args())
