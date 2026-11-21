"""Two complementary science observers with explicit component cost accounting."""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import numpy as np

import continuous_repairability as policy
from subject_science_pomdp import SubjectPlanner


def paid_prefix(tree, path):
    paths = ['']
    if path:
        parts = path.split('/')
        paths.extend('/'.join(parts[:i]) for i in range(1, len(parts) + 1))
    return {p: tree[p] for p in paths}


@dataclass
class FusionObserver:
    components: list
    _recipe: dict = field(default_factory=dict, repr=False)
    _asset_ids: dict = field(default_factory=dict, repr=False)

    def __getstate__(self):
        return {**vars(self), '_recipe': {}, '_asset_ids': {}}

    @property
    def device(self):
        return self.components[0].device

    @device.setter
    def device(self, value):
        for component in self.components:
            component.device = value

    def load(self):
        if len(self.components) != 2:
            raise ValueError('Use exactly the rationale and option science observers')
        for component in self.components:
            component.load()
        parameters = [c._recipe['parameters'] for c in self.components]
        lengths = [c._recipe['max_length'] for c in self.components]
        profiles = [c._recipe['input_profile'] for c in self.components]
        if parameters[0] != parameters[1] or lengths != [512, 512]:
            raise ValueError('This fusion billing requires equal per-pass parameters and two512caps')
        if profiles != ['full-options-openstax-v1', 'supergpqa-option-correctness-v1']:
            raise ValueError('Incorrect fusion observer profiles or ordering')
        # The physical total is honest: both sets of weights are resident.
        # Core fitting uses parameters * max_length, equal to the sum of both
        # conservative512-slot component charges. Runtime bills actual slots.
        self._recipe = {'parameters': sum(parameters), 'max_length': 512,
                        'input_profile': 'rationale-plus-auxiliary-option-fusion-v1',
                        'physical_parameters': sum(parameters),
                        'per_forward_parameters': parameters[0], 'forward_passes_per_observation': 2,
                        'maximum_total_slots_per_observation': sum(lengths),
                        'component_parameters': parameters, 'component_max_length': lengths,
                        'runtime_billing': 'Sum(component_parameters * actual_component_slots) /3.09e9'}

    def values(self, row, path, tree, judge=None):
        if row['subject'] not in ('chemistry', 'physics'):
            raise ValueError('Science fusion cannot silently add an encoder to math')
        self.load()
        safe, paid = {**row, 'answer': ''}, paid_prefix(tree, path)
        values = [c.values(safe, path, paid, judge) for c in self.components]
        return (*[np.concatenate([np.asarray(v[i]) for v in values]) for i in (0, 1, 2)],
                sum(int(v[3]) for v in values))

    def component_charges(self, row, tree, judges, decisions, include_reviews=True):
        self.load()
        result = []
        for index, component in enumerate(self.components):
            directory = str(Path(component.directory).resolve())
            if directory not in self._asset_ids:
                self._asset_ids[directory] = hashlib.sha256((Path(directory) / 'recipe.json').read_bytes()).hexdigest()
            parameters = component._recipe['parameters']
            for decision in decisions:
                path = decision['path']; safe, paid = {**row, 'answer': ''}, paid_prefix(tree, path)
                passes = [('solver', None)]
                if decision['queried'] and include_reviews:
                    passes.append(('review', judges[row['id'], path]))
                for kind, judge in passes:
                    slots = int(component.values(safe, path, paid, judge)[3])
                    result.append({'component': index, 'profile': component._recipe['input_profile'],
                                   'recipe_sha256': self._asset_ids[directory], 'kind': kind, 'path': path,
                                   'parameters': parameters, 'token_slots': slots, 'calls': 1,
                                   'compute': parameters / 1e9 * slots / 3.09})
        return result

    def charges(self, row, tree, judges, decisions, include_reviews=True):
        components = self.component_charges(row, tree, judges, decisions, include_reviews)
        # Equal per-forward parameter count times the sum of actual slots is
        # algebraically the componentwise charge. This is not the total number
        # of resident model weights, which is separately recorded above/below.
        slots = sum(c['token_slots'] for c in components)
        calls = sum(c['calls'] for c in components)
        return slots, calls, slots, self._recipe['per_forward_parameters'] / 1e9

    def provenance(self):
        self.load()
        return {**self._recipe, 'components': [
            {'directory': str(Path(c.directory).resolve()), 'profile': c._recipe['input_profile'],
             'parameters': c._recipe['parameters'],
             'recipe_sha256': hashlib.sha256((Path(c.directory) / 'recipe.json').read_bytes()).hexdigest()}
            for c in self.components]}


def evaluate(row, tree, judges, model, settings, family):
    if family != 'A':
        raise ValueError('The science-fusion POMDP has no LLM-review action')
    submodel = model.models[row['subject']]
    item = policy.evaluate(row, tree, {}, submodel, model.settings[row['subject']], 'A')
    observer = submodel.classifier
    if isinstance(observer, FusionObserver):
        ledger = observer.component_charges(row, tree, {}, item['decisions'], include_reviews=False)
        charged = item['verifier_parameters_billions'] * item['verifier_token_slots'] / 3.09
        if abs(charged - sum(c['compute'] for c in ledger)) > 1e-8:
            raise ValueError('Aggregate and component encoder charges differ')
        item['verifier_components'] = ledger
        item['verifier_physical_parameters_billions'] = observer._recipe['physical_parameters'] / 1e9
        item['verifier_billing_parameters_billions_per_pass'] = observer._recipe['per_forward_parameters'] / 1e9
    return item
