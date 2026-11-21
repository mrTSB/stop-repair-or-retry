"""Frozen semantic-policy dispatch and receipts for a single unused confirmation panel."""

import importlib
import json
from pathlib import Path

import numpy as np
import torch

import semantic_data
from study_common import (BASELINES, load_test, paired_margins, sha, statistics,
                          evidence as original_evidence, models as original_models)
from context_pomdp import evaluate_llm
from late_pairs import llm_policy


def evidence(data):
    result = original_evidence(data)
    for p in sorted(semantic_data.HERE.glob('*.py')):
        result['semantic-code/' + p.name] = sha(p)
    for name in ('manifest.json', 'passages.jsonl', 'tfidf.pkl'):
        p = semantic_data.ROOT / '.cache/semantic-textbooks' / name
        result['textbook/' + name] = sha(p)
    return result


def models(frozen, output):
    result = original_models(frozen, output)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    for model in result.values():
        model.classifier.device = device
    cache = output / 'test-neural-cache.jsonl'
    if cache.exists():
        seen = {}
        for line in cache.read_text().splitlines():
            item = json.loads(line)
            key = item['family'], tuple(item['key'])
            if key in seen and seen[key] != item:
                raise ValueError('Changed cached neural observation')
            seen[key] = item
            if item['family'] not in result:
                raise ValueError('Unknown observation-model family')
            descriptor = frozen['models'][item['family']]
            recipe = output / descriptor['observer_directory'] / 'recipe.json'
            if item['recipe_sha256'] != sha(recipe):
                raise ValueError('Wrong cached observation weights/profile')
            result[item['family']].classifier._cache[tuple(item['key'])] = (
                np.asarray(item['state'], dtype=np.float32), np.asarray(item['plain'], dtype=np.float32),
                np.asarray(item['guided'], dtype=np.float32), item['tokens'])
    return result


def run_policy(family, row, tree, base_reviews, late_reviews, model, configuration):
    reviews = late_reviews if configuration['protocol'] == 'late' else base_reviews
    if family == 'B':
        if model is not None:
            raise ValueError('An LLM-only controller must not receive a POMDP')
        return (llm_policy if configuration['protocol'] == 'late' else evaluate_llm)(
            row, tree, reviews, configuration['settings'])
    module = importlib.import_module(configuration['module'])
    return module.evaluate(row, tree, reviews, model, configuration['settings'], family)
