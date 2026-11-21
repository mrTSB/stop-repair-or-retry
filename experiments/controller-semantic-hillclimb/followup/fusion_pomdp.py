"""POMDP observation fusion of two existing discriminative encoders.

Both encoders process the same tokenized observation. Their question-excluded
logits fit a joint correctness/difficulty calibration; neither generates text.
Both physical forwards and all their token slots are charged.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dynamic_grounded_pomdp as perception
from evidence import Retriever, DEFAULT as CORPUS
from network import load


class FusionNetwork(nn.Module):
    def __init__(self, components, temperatures):
        super().__init__()
        self.components = nn.ModuleList(components)
        self.temperatures = temperatures

    def forward(self, **kw):
        outputs = [model(**kw) for model in self.components]
        return (torch.cat([value[0] / temperature for value, temperature in zip(outputs, self.temperatures)], dim=-1),
                torch.stack([value[1] for value in outputs]).mean(0),
                torch.stack([value[2] for value in outputs]).mean(0))


@dataclass
class Observer(perception.Observer):
    def load(self):
        if self._model is not None:
            return
        torch.set_num_threads(4)
        directory = Path(self.directory)
        self._recipe = json.loads((directory / 'recipe.json').read_bytes())
        if self._recipe['kind'] != 'parallel-observer-fusion-v1':
            raise ValueError('Unexpected fusion recipe')
        components, tokenizers, temperatures = [], [], []
        fingerprints = set()
        for component in self._recipe['components']:
            folder = directory / component['directory']
            model, tokenizer, recipe = load(folder)
            if recipe['input_profile'] != 'full-options-openstax-v1':
                raise ValueError('Incompatible observation input profile')
            fingerprints.add(hashlib.sha256((folder / 'spm.model').read_bytes()).hexdigest())
            components.append(model)
            tokenizers.append(tokenizer)
            temperatures.append(component['temperature'])
        if len(fingerprints) != 1 or any(t.all_special_ids != tokenizers[0].all_special_ids for t in tokenizers):
            raise ValueError('Fusion requires identical tokenization, including special tokens')
        self._model = FusionNetwork(components, temperatures).to(self.device)
        self._tokenizer = tokenizers[0]
        if sum(p.numel() for p in self._model.parameters()) != self._recipe['parameters']:
            raise ValueError('Incorrect fusion parameter accounting')
        self._retriever = self._retriever or Retriever(CORPUS)
        if self._retriever.sha256 != self._recipe['corpus_sha256']:
            raise ValueError('Changed textbook evidence')

    def charges(self, row, tree, judges, decisions, include_reviews=True):
        tokens, calls, slots, total_parameters = super().charges(row, tree, judges, decisions, include_reviews)
        components = self._recipe['component_count']
        # The average parameter price times all physical slots is exactly the
        # sum of each component's parameter price times its identical input.
        return tokens * components, calls * components, slots * components, total_parameters / components


def fit(rows, trees, judges, directory, metadata, outputs, smoothing=32., late=True,
        semantic_review=False, context_weight=0.):
    if semantic_review:
        raise ValueError('This trial fuses solver observations and retains ordinary paid LLM signals')
    initial = perception.fit(rows, trees, judges, directory, metadata, outputs,
                             smoothing=smoothing, late=late, semantic_review=False,
                             context_weight=context_weight)
    initial.classifier = Observer(**vars(initial.classifier))
    initial.review_observer = None
    initial.fit_receipt.update(observer_fusion=True, physical_encoder_components=2,
                               joint_calibration='Concatenated question-excluded logits from both encoders',
                               neural_training='No additional neural fitting; reuse the two existing grouped-fold models',
                               physical_accounting='Every component forward and token slot is charged; effective mean parameter price is accompanied by total parameters')
    return initial


def evaluate(row, tree, judges, model, settings, family):
    item = perception.evaluate(row, tree, judges, model, settings, family)
    item['verifier_total_parameters_billions'] = model.classifier._recipe['parameters'] / 1e9
    item['verifier_components'] = model.classifier._recipe['component_count']
    return item
