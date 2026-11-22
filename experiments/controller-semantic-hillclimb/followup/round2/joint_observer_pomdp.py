"""Paid-prefix joint observer feeding the existing two-step science POMDP.

This adapter is preparation only. A trained model must pass its separate
registered screen before a full fit or development run may use this adapter.
The input builder supplies text/numeric features, never its old neural logits.
"""
import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import fit_expanded1280_completion as source
import joint_outcome_observer as network
import continuous_completion_pomdp as policy
import clustered_completion_pomdp as clustered

io, fixed, runtime = source.io, source.fixed, source.runtime
core = runtime.core


@dataclass
class Observer:
    directory: str
    input_builder: object
    recipe_sha256: str
    device: str = 'cpu'
    _recipe: dict = field(default_factory=dict, repr=False)
    _model: object = field(default=None, repr=False)
    _tokenizer: object = field(default=None, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def __getstate__(self):
        return {**vars(self), '_model': None, '_tokenizer': None, '_cache': {}}

    def metadata(self):
        path = Path(self.directory)/'recipe.json'
        if io.sha(path) != self.recipe_sha256:
            raise ValueError('Joint observer recipe changed')
        recipe = io.read(path)
        if recipe['architecture'] != network.ARCHITECTURE:
            raise ValueError('Wrong observer architecture')
        if recipe['initial_directory'] != str(Path(self.input_builder.directory).resolve()):
            raise ValueError('Wrong paid-prefix input builder')
        if io.sha(Path(self.input_builder.directory)/'recipe.json') != recipe['initial_recipe_sha256']:
            raise ValueError('Paid-prefix input recipe changed')
        self._recipe = recipe
        return recipe

    def entry(self, row, path, paid):
        if row.get('answer') != '' or path not in ('', 'repair', 'retry'):
            raise ValueError('Only blank-reference nonterminal paid inputs')
        if set(paid) != {'', path}:
            raise ValueError('Unpaid or missing response in joint observation')
        result = self.input_builder.entry(row, path, paid)
        if self.input_builder._retriever.sha256 != self.metadata()['corpus_sha256']:
            raise ValueError('Paid-prefix corpus changed')
        return {k: result[k] for k in ('left', 'right', 'numeric')}

    def values(self, row, path, paid):
        entry = self.entry(row, path, paid)
        key = fixed.fingerprint({'entry': entry, 'valid': bool(paid[path]['answer'])})
        if key not in self._cache:
            recipe = self.metadata()
            if self._model is None:
                self._model, self._tokenizer, loaded = network.load(self.directory)
                if loaded != recipe:
                    raise ValueError('Observer changed during loading')
                self._model = self._model.to(self.device)
            tensors = network.initial.encode(self._tokenizer, [entry], recipe['max_length'])
            logits = network.predict(self._model, tensors, self.device, batch=1)
            p = network.probabilities(logits, [bool(paid[path]['answer'])], recipe['prior'])[0]
            self._cache[key] = (p, int(tensors['attention_mask'].sum()))
        return self._cache[key]

    def charges(self, row, paid, judges, decisions, include_reviews=False):
        if judges or include_reviews or any(d.get('queried', False) for d in decisions):
            raise ValueError('Joint observer A never requests an LLM')
        paths = [d['path'] for d in decisions]
        if len(set(paths)) != len(paths) or not paths or paths[0] != '':
            raise ValueError('Exactly one observation per visited decision prefix')
        slots = sum(self.values(row, p, {k: paid[k] for k in ('', p)})[1] for p in paths)
        return slots, len(paths), slots, self.metadata()['parameters']/1e9


@dataclass
class OriginalKernel:
    def signal(self, probability, valid):
        return core.observation(probability, valid)


@dataclass
class Planner:
    classifier: object
    dynamics: dict
    fit_receipt: dict
    kernel: object

    def predicted(self, row, path, tree):
        if row['subject'] not in ('chemistry', 'physics'):
            raise ValueError('Joint science observer does not replace the math policy')
        paid = {p: tree[p] for p in ('', path)}
        p, _ = self.classifier.values({**row, 'answer': ''}, path, paid)
        return core.mask_invalid(p, bool(paid[path]['answer']))


def fit(rows, trees, observer, records, oof, provenance):
    observer.metadata()
    original = runtime.fit_dynamics(rows, trees, observer, records, oof, None,
        'raw20_control', provenance)
    variants = {'original': (original, OriginalKernel())}
    alternative = clustered.rebin(original, rows, trees, records, oof)
    variants['clustered'] = alternative, alternative.kernel
    result = {}
    for name, (model, kernel) in variants.items():
        receipt = {**model.fit_receipt, 'schema': 'direct-joint-observer-Complete16-POMDP-v1',
            'feature_family': 'direct-joint16-paid-prefix',
            'observation_forecasts': 'Fixed-recipe whole-question excluded observer predictions',
            'observer_recipe_sha256': observer.recipe_sha256,
            'discriminative_observer_never_generates_answers': True,
            'original_solver_action_protocols_unchanged': True}
        result[name] = Planner(copy.deepcopy(observer), model.dynamics, receipt, kernel)
    return result


evaluate_science = policy.evaluate_science
evaluate = policy.evaluate
SubjectPlanner = policy.SubjectPlanner
