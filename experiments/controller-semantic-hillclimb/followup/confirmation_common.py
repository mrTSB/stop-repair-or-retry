"""Frozen912study dispatch, reference-free observations, and source-stratified margins."""

import hashlib
import importlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import semantic_data
from semantic_data import read
from semantic_study_common import evidence as previous_evidence
from study_common import BASELINES, statistics as ordinary_statistics
from context_pomdp import evaluate_llm
from late_pairs import llm_policy
from banks import ordinary, medium_view, committed

RUNTIME = ('banks.py', 'binary_core.py', 'binary_pomdp.py', 'calibrated_pomdp.py', 'medium_pomdp.py',
           'root_bank.py', 'root_pairs.py', 'root_medium_pomdp.py', 'fusion_pomdp.py', 'subject_fusion_pomdp.py',
           'mentor_pomdp.py', 'reuse_pairs.py',
           'medium_late_pairs.py', 'transfer_variants.py', 'confirmation_common.py',
           'confirmation.py', 'confirmation_collect.py', 'confirmation_audit.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_fixed_prefix(frozen, data):
    content = (data / 'test-observations.jsonl').read_bytes()
    prefix = content[:frozen['fixed_cache_bank_bytes']]
    if len(prefix) != frozen['fixed_cache_bank_bytes'] or hashlib.sha256(prefix).hexdigest() != frozen['fixed_cache_bank_sha256']:
        raise ValueError('The prespecified fixed baseline response cache changed after freeze')


def verify_special_banks(frozen, medium):
    if Path(frozen['medium_directory']).resolve() != medium.resolve():
        raise ValueError('Use the selected and frozen hybrid response bank')
    for suffix, name in (('contract', 'contract.json'), ('bank', 'train.jsonl')):
        if sha(medium / name) != frozen['medium_training_' + suffix + '_sha256']:
            raise ValueError('Changed frozen hybrid training evidence')
    expected = frozen['medium_development_bank_sha256']
    if expected is not None and sha(medium / 'dev.jsonl') != expected:
        raise ValueError('Changed frozen hybrid development evidence')
    for name, digest in frozen.get('reuse_bank', {}).get('files', {}).items():
        if sha(Path(frozen['reuse_bank']['directory']) / name) != digest:
            raise ValueError('Changed frozen mentor-reuse evidence')


def evidence(data):
    result = previous_evidence(data)
    result.update({'followup-code/' + name: sha(HERE / name) for name in RUNTIME})
    return result


def load_banks(data, split, medium, protocol='medium', reuse=None):
    rows, trees, original, late = ordinary(data, split)
    if protocol == 'mentor-reuse':
        from mentor_pomdp import reuse_view
        if reuse is None:
            raise ValueError('Supply the frozen mentor-reuse bank')
        deeper, reviews = reuse_view(rows, trees, committed(medium / (split + '.jsonl')),
                                     committed(reuse / (split + '.jsonl')), validate=True)
    elif protocol == 'root-medium':
        from root_bank import view
        deeper, reviews = view(rows, trees, committed(medium / (split + '.jsonl')), validate=True)
    elif protocol == 'medium':
        deeper, reviews = medium_view(rows, trees, original, committed(medium / (split + '.jsonl')), validate=True)
    else:
        deeper, reviews = trees, late
    return rows, trees, original, late, deeper, reviews


def clear(model):
    if model is not None:
        for attr in ('_observed', '_current_context', '_review_context'):
            getattr(model, attr, {}).clear()
        getattr(model.classifier, '_probability_cache', {}).clear()
        if hasattr(model, 'regular'):
            clear(model.regular)


def models(frozen, output, load_cache=True):
    result = {}
    shared = {}
    for family, descriptor in frozen['models'].items():
        path = output / descriptor['file']
        if sha(path) != descriptor['sha256']:
            raise ValueError('Changed frozen planner')
        folder = output / descriptor['observer_directory']
        for name, digest in frozen['observer_assets'][descriptor['observer']]['files'].items():
            if sha(folder / name) != digest:
                raise ValueError('Changed observer asset')
        model = pickle.loads(path.read_bytes())
        model.classifier.directory = str(folder.resolve())
        model.classifier.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        observer_key = descriptor['observer']
        if observer_key in shared:
            observer = shared[observer_key]
            for attr in ('_model', '_tokenizer', '_recipe', '_cache', 'device', '_retriever', '_backgrounds'):
                setattr(model.classifier, attr, getattr(observer, attr))
        else:
            model.classifier.load()
            shared[observer_key] = model.classifier
        result[family] = model
    cache = output / 'test-neural-cache.jsonl'
    if load_cache and cache.exists():
        seen = {}
        for item in committed(cache):
            family = item['family']
            key = family, tuple(item['key'])
            if key in seen and seen[key] != item:
                raise ValueError('Changed duplicate neural receipt')
            seen[key] = item
            descriptor = frozen['models'][family]
            if item['recipe_sha256'] != sha(output / descriptor['observer_directory'] / 'recipe.json'):
                raise ValueError('Wrong cached model weights/profile')
            result[family].classifier._cache[tuple(item['key'])] = (
                *[np.asarray(item[k], dtype=np.float32) for k in ('state', 'plain', 'guided')],
                int(item['tokens']))
    return result


def run_policy(family, row, tree, original, late, deeper, deeper_reviews, model, configuration):
    protocol = configuration['protocol']
    if family == 'B':
        if model is not None or protocol in ('medium', 'root-medium', 'mentor-reuse'):
            raise ValueError('Strong ordinary B stays an independent LLM-only reference')
        function = llm_policy if protocol == 'late' else evaluate_llm
        reviews = late if protocol == 'late' else original
        result = function(row, tree, reviews, configuration['settings'])
    else:
        module = importlib.import_module(configuration['module'])
        special = protocol in ('medium', 'root-medium', 'mentor-reuse')
        reviews = deeper_reviews if special else late if protocol == 'late' else original
        result = module.evaluate(row, deeper if special else tree, reviews,
                                 model, configuration['settings'], family)
    result['judge_physical_calls'] = sum(reviews[row['id'], d['path']].get('physical_api_calls', 1)
                                         for d in result['decisions'] if d['queried'])
    return result


def statistics(items):
    return {**ordinary_statistics(items),
            'mean_judge_physical_calls': float(np.mean([r.get('judge_physical_calls', 0) for r in items]))}


def margins(rows, results, panel_by_id, repetitions=5000):
    names = list(results)
    accuracy = np.array([[r['correct'] for r in results[k]] for k in names], dtype=float)
    cost = np.array([[r['compute'] for r in results[k]] for k in names], dtype=float)
    baseline = [names.index(k) for k in BASELINES]
    a, b, c = [names.index(k) for k in ('A', 'B', 'C')]

    def quantities(acc, work):
        highest = acc[baseline].max(axis=0)
        return {'A_above_highest_baseline': acc[a] - highest,
                'B_above_highest_baseline': acc[b] - highest,
                'C_above_highest_A_B': acc[c] - np.maximum(acc[a], acc[b]),
                'C_above_B': acc[c] - acc[b],
                'C_compute_saved_vs_B': work[b] - work[c],
                'C_compute_fraction_saved_vs_B': 1 - work[c] / work[b]}

    point = quantities(accuracy.mean(1), cost.mean(1))
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((panel_by_id[row['id']], row['subject']), []).append(i)
    rng = np.random.default_rng(2026112063)
    draws = np.concatenate([rng.choice(g, (repetitions, len(g)), replace=True)
                            for _, g in sorted(groups.items())], axis=1)
    boot = quantities(accuracy[:, draws].mean(2), cost[:, draws].mean(2))
    return {k: {'point': float(v), 'ci95': np.quantile(boot[k], [.025, .975]).tolist()}
            for k, v in point.items()}


def criteria(measured):
    return {'A_plus_10_points': measured['A_above_highest_baseline']['point'] >= .1 - 1e-12,
            'B_plus_10_points': measured['B_above_highest_baseline']['point'] >= .1 - 1e-12,
            'C_plus_10_points': measured['C_above_highest_A_B']['point'] >= .1 - 1e-12,
            'C_plus_15_points_stretch': measured['C_above_highest_A_B']['point'] >= .15 - 1e-12,
            'C_accuracy_above_B': measured['C_above_B']['point'] > 0,
            'C_compute_below_B': measured['C_compute_saved_vs_B']['point'] > 0,
            'C_accuracy_clearly_above_B': measured['C_above_B']['ci95'][0] > 0,
            'C_compute_clearly_below_B': measured['C_compute_saved_vs_B']['ci95'][0] > 0}
