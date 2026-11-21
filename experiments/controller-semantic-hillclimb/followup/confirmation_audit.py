"""Independently reconcile charges and replay only paid inputs with hidden references."""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from confirmation_common import (BASELINES, evidence, load_banks, models, run_policy, verify_fixed_prefix, verify_special_banks,
                                 clear, read, sha, statistics)
from banks import ordinary, variant_observations, medium_view
from medium_pomdp import merged_observations
from study import fixed
from scoring import correct


def refit(configuration, frozen_model, data, medium, examples, variants, supplement):
    rows, trees, original, late = ordinary(data, 'train')
    folder = Path(frozen_model.classifier.directory)
    # The archived observer weights are identical to the original full model.
    # The source path locates its independent training-only OOF predictions.
    original_folder = Path(frozen_model._source_observer_directory)
    metadata = read(examples / 'train.jsonl')
    outputs = np.load(original_folder.parent / 'oof.npz')
    module = importlib.import_module(configuration['module'])
    kwargs = {'smoothing': frozen_model.fit_receipt['smoothing'],
              'semantic_review': frozen_model.review_observer is not None}
    if configuration['protocol'] in ('root-medium', 'mentor-reuse', 'full-packet'):
        from root_bank import view
        from banks import committed
        if configuration['protocol'] == 'full-packet':
            trees, reviews = module.view(rows, trees, original, committed(medium / 'train.jsonl'), validate=True)
            training_observations = module.training_observations
        elif configuration['protocol'] == 'mentor-reuse':
            from mentor_pomdp import reuse_view
            trees, reviews = reuse_view(rows, trees, committed(medium / 'train.jsonl'),
                                        committed(Path(configuration['reuse_directory']) / 'train.jsonl'), validate=True)
            from root_medium_pomdp import training_observations
        else:
            trees, reviews = view(rows, trees, committed(medium / 'train.jsonl'), validate=True)
            training_observations = module.training_observations
        metadata, outputs = training_observations(metadata, outputs, read(supplement / 'metadata.jsonl'),
                                                         np.load(supplement / 'oof.npz'))
    elif configuration['protocol'] == 'medium':
        observations = variant_observations(medium, variants, 'medium-short')
        trees, reviews = medium_view(rows, trees, original, observations, validate=True)
        metadata, outputs = merged_observations(metadata, outputs, read(supplement / 'metadata.jsonl'),
                                                np.load(supplement / 'oof.npz'))
        kwargs['state_count'] = frozen_model.fit_receipt['state_count']
    else:
        reviews = late if configuration['protocol'] == 'late' else original
        kwargs.update(late=configuration['protocol'] == 'late', context_weight=0.)
    rebuilt = module.fit(rows, trees, reviews, original_folder, metadata, outputs, **kwargs)
    rebuilt.classifier.directory = str(folder)
    for attr in ('_model', '_tokenizer', '_recipe', '_cache', 'device', '_retriever', '_backgrounds'):
        setattr(rebuilt.classifier, attr, getattr(frozen_model.classifier, attr))
    return rebuilt


def run(args):
    frozen = json.loads((args.output / 'frozen.json').read_bytes())
    if frozen['evidence'] != evidence(args.data):
        raise ValueError('Frozen sources/development inputs changed')
    verify_fixed_prefix(frozen, args.data)
    verify_special_banks(frozen, args.medium)
    reuse = Path(frozen['reuse_bank']['directory']) if 'reuse_bank' in frozen else None
    if reuse is not None:
        for name, digest in frozen['reuse_bank']['files'].items():
            assert sha(reuse / name) == digest
    rows, trees, original, late, deeper, deep_reviews = load_banks(
        args.data, args.split, args.medium, frozen['selected']['C']['protocol'], reuse)
    planners = models(frozen, args.output, load_cache=args.split == 'test')
    scored = None
    if args.split == 'test':
        scored = {(r['policy'], r['id']): r for r in read(args.output / 'test/per-question.jsonl')}
        if len(scored) != len(rows) * 8:
            raise ValueError('Incomplete scored result set')
    for model in planners.values():
        for d in model.dynamics.values():
            for name in ('prior', 'initial', 'transition', 'emission', 'review', 'reviewed_transition',
                         'reviewed_emission', 'mentor', 'last_transition'):
                if not hasattr(d, name):
                    continue
                array = getattr(d, name)
                assert np.isfinite(array).all() and (array >= 0).all()
                assert np.allclose(array.sum(-1), 1.), name
    rebuilt = {}
    if args.split == 'dev' and args.refit:
        for family, model in planners.items():
            descriptor = frozen['models'][family]
            model._source_observer_directory = frozen['observer_assets'][descriptor['observer']]['source_directory']
            rebuilt[family] = refit(frozen['selected'][family], model, args.data, args.medium,
                                    args.examples, args.variants, args.supplement)
    outcomes = []
    charge_checks = blind_checks = refit_checks = regeneration_checks = 0
    for family in (*BASELINES, 'A', 'B', 'C'):
        model = planners.get(family)
        config = frozen['selected'].get(family)
        is_medium = config is not None and config['protocol'] in ('medium', 'root-medium', 'mentor-reuse', 'full-packet')
        bank = deeper if is_medium else trees
        reviews = deep_reviews if is_medium else late if config and config['protocol'] == 'late' else original
        items = []

        def policy(row, tree, paid_reviews, current=model):
            if family in BASELINES:
                return fixed(row, tree, *BASELINES[family])
            clear(current)
            return run_policy(family, row, tree, paid_reviews, paid_reviews, tree, paid_reviews, current, config)

        for row_index, row in enumerate(rows):
            item = policy(row, bank[row['id']], reviews)
            if scored is not None:
                expected = dict(scored[family, row['id']])
                expected.pop('policy'); expected.pop('panel')
                assert item == expected, ('saved outcome', family, row['id'])
            items.append(item)
            parts = item['path'].split('/') if item['path'] else []
            visited = [''] + ['/'.join(parts[:i]) for i in range(1, len(parts) + 1)]
            assert len(visited) <= 3
            decisions = item.get('decisions', [])
            acquired = [d['path'] for d in decisions if d['queried']]
            assert len(acquired) == len(set(acquired)) <= 2
            assert set(acquired) <= set(visited)
            if family == 'A':
                assert acquired == []
            if model is not None and model.fit_receipt.get('maximum_logical_reviews') == 1:
                assert len(acquired) <= 1 and set(acquired) <= {''}
            paid_tree = {p: bank[row['id']][p] for p in visited}
            paid_reviews = {(row['id'], p): reviews[row['id'], p] for p in acquired}
            solver = sum(n['input_tokens'] + n['output_tokens'] for n in paid_tree.values())
            judge = sum(n['input_tokens'] + n['output_tokens'] for n in paid_reviews.values())
            physical = 0
            for node in paid_reviews.values():
                physical += node.get('physical_api_calls', 1)
                if 'calls' in node:
                    assert len(node['calls']) == node['physical_api_calls']
                    for field in ('input_tokens', 'output_tokens'):
                        assert sum(c[field] for c in node['calls']) == node[field]
            assert (solver, judge, len(visited), len(acquired)) == (
                item['solver_tokens'], item['judge_tokens'], item['solver_calls'], item['judge_calls'])
            assert item.get('judge_physical_calls', 0) == physical
            encoder = 0.
            if model is not None:
                logical_calls = len(decisions) + (len(acquired) if model.review_observer is not None else 0)
                components = model.classifier._recipe.get('component_count', 1)
                expected_calls = logical_calls * components
                expected_tokens = sum(model.classifier.values(row, d['path'], paid_tree)[3] for d in decisions)
                if model.review_observer is not None:
                    expected_tokens += sum(model.classifier.values(row, p, paid_tree, n)[3]
                                           for (_, p), n in paid_reviews.items())
                assert item['verifier_calls'] == expected_calls
                assert item['verifier_tokens'] == item['verifier_token_slots'] == expected_tokens * components
                encoder = model.classifier._recipe['parameters'] / 1e9 * expected_tokens
                assert np.isclose(item['verifier_parameters_billions'], model.classifier._recipe['parameters'] / 1e9 / components)
                if components > 1:
                    assert item['verifier_total_parameters_billions'] == model.classifier._recipe['parameters'] / 1e9
                state_count = len(next(iter(model.dynamics.values())).prior)
                assert state_count in (2, 4, 8)
                for d in decisions:
                    assert len(d['belief']) == state_count
                    assert np.isclose(sum(d['belief']), 1.) and min(d['belief']) >= 0
            assert np.isclose(item['compute'], (4 * solver + 3 * judge + encoder) / 3.09)
            assert np.isclose(item.get('total_parameter_compute', item['compute']),
                              (4 * solver + 35 * judge + encoder) / 3.09)
            assert item['correct'] == int(correct(paid_tree[item['path']], row['answer']))
            charge_checks += 1
            if family not in BASELINES:
                expected = {k: v for k, v in item.items() if k != 'correct'}
                for replacement in ('', '918273645546372819'):
                    actual = policy({**row, 'answer': replacement}, paid_tree, paid_reviews)
                    actual.pop('correct')
                    assert actual == expected, (family, row['id'])
                    blind_checks += 1
                if family in rebuilt:
                    actual = policy(row, paid_tree, paid_reviews, rebuilt[family])
                    assert actual == item, ('refit', family, row['id'])
                    refit_checks += 1
            if model is not None and row_index < args.regenerate:
                descriptors = [(d['path'], None) for d in decisions]
                if model.review_observer is not None:
                    descriptors += [(p, n) for (_, p), n in paid_reviews.items()]
                for path, review in descriptors:
                    node = paid_tree[path] if review is None else review
                    key = model.classifier.key(row, path, node, 'solver' if review is None else 'review')
                    expected = model.classifier._cache.pop(key)
                    actual = model.classifier.values({**row, 'answer': ''}, path, paid_tree, review)
                    assert actual[3] == expected[3]
                    assert all(np.allclose(x, y, atol=1e-4, rtol=1e-4) for x, y in zip(actual[:3], expected[:3]))
                    regeneration_checks += 1
            outcomes.append({'policy': family, **item})
        measured = statistics(items)
        if args.split == 'dev' and family not in BASELINES:
            expected = config['dev']
            assert all(np.isclose(measured[k], v, atol=1e-7, rtol=1e-7) for k, v in expected.items()), (family, measured, expected)
        print(f'Audited {family}: {len(items)} independent charges and paid-prefix replays', flush=True)
    receipt = {'passed': True, 'split': args.split, 'questions': len(rows),
               'freeze_sha256': sha(args.output / 'frozen.json'), 'independent_charge_checks': charge_checks,
               'blind_pruned_replays': blind_checks, 'training_only_refitted_outcomes': refit_checks,
               'independent_encoder_regenerations': regeneration_checks}
    if args.split == 'test':
        receipt.update(scored_summary_sha256=sha(args.output / 'test/summary.json'),
                       scored_outcomes_sha256=sha(args.output / 'test/per-question.jsonl'))
    name = 'development-audit.json' if args.split == 'dev' else 'test-audit.json'
    (args.output / name).write_text(json.dumps(receipt, indent=2) + '\n')
    if args.split == 'dev':
        with (args.output / 'audited-development.jsonl').open('w') as stream:
            for item in outcomes:
                stream.write(json.dumps(item) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'medium', 'output', 'examples', 'variants', 'supplement'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--split', choices=('dev', 'test'), default='dev')
    p.add_argument('--refit', action='store_true')
    p.add_argument('--regenerate', type=int, default=12)
    run(p.parse_args())
