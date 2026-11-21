"""Paid, answer-only science observations with an explicit auxiliary profile.

This observer has binary correctness supervision only. It supplies neutral
difficulty and transition information; a downstream POMDP must fit those from
its own paired training data. Each requested answer costs one actual, unpadded
encoder forward pass, including an invalid answer or acquired LLM review.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import banks
from dynamic_grounded_pomdp import Observer, trim_tensors
from network_large import encode, load, predict
from science_auxiliary_data import PROFILE, option_entry


def binary_outputs(state):
    """Retain only the trained binary margin, for bulk and online inference."""
    state = np.asarray(state)
    logit = np.logaddexp(state[..., 1], state[..., 3]) - np.logaddexp(state[..., 0], state[..., 2])
    zeros = np.zeros_like(logit)
    return np.stack((zeros, logit, zeros, logit), axis=-1)


@dataclass
class ScienceOptionObserver(Observer):
    def load(self):
        if self._model is None:
            torch.set_num_threads(4)
            self._model, self._tokenizer, self._recipe = load(self.directory)
            if self._recipe.get('input_profile') != PROFILE:
                raise ValueError('Wrong science option observation input profile')
            if self._recipe.get('transition_heads') != 'Frozen and not supervised; online observer returns neutral transition logits':
                raise ValueError('Ambiguous auxiliary transition-head provenance')
            self._model = self._model.to(self.device)

    def entry(self, row, path, tree, judge=None):
        if row['subject'] not in ('chemistry', 'physics'):
            raise ValueError('The auxiliary option observer is science-only')
        if not self._recipe:
            self.load()
        node = tree[path] if judge is None else judge
        # option_entry never reads the reference answer or the solver rationale.
        return option_entry(row, node['answer'], self._recipe['numeric_width'])

    def values(self, row, path, tree, judge=None):
        self.load()
        entry = self.entry(row, path, tree, judge)
        fingerprint = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        key = ('solver' if judge is None else 'review', row['id'], path, fingerprint)
        if key not in self._cache:
            tensors = trim_tensors(encode(self._tokenizer, [entry], self._recipe['max_length']))
            state = predict(self._model, tensors, self.device, batch=1)[0][0]
            # Remove arbitrary legacy hidden-difficulty splits: only this
            # correct-vs-incorrect margin received auxiliary supervision.
            self._cache[key] = (binary_outputs(state), np.zeros(8), np.zeros(8),
                                int(tensors['attention_mask'].sum()))
        return self._cache[key]


def from_checkpoint(directory, device='cpu'):
    directory = Path(directory)
    recipe = json.loads((directory / 'recipe.json').read_bytes())
    return ScienceOptionObserver(str(directory.resolve()), float(recipe['temperature']), device=device)
