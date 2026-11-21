"""Paid sparse-word observations for the unchanged eight-state finite POMDP."""
import copy
from dataclasses import dataclass, field
import time

import numpy as np

import repairability_pomdp as core
import continuous_repairability as continuous
import sparse_text_repairability as text


@dataclass
class Planner(core.Planner):
    head_receipts: list = field(default_factory=list, repr=False)

    def predicted(self, row, path, tree):
        safe = {**row, 'answer': ''}
        paid = {p: tree[p] for p in ('', path)}
        neural = self.classifier.values(safe, path, paid)
        started = time.perf_counter()
        numeric = core.observables(safe, path, paid, neural)
        words = text.paid_text(safe, path, paid)
        p = self.forecaster.predict_proba([words], [numeric])[0]
        if not paid[path]['answer']:
            p[core.CURRENT == 1] = 0.
            p = core.normalize(p)
        self.head_receipts.append({'path': path, 'input_sha256': text.fingerprint([words, numeric]),
                                   'cpu_seconds': time.perf_counter() - started, 'neural_calls': 0})
        return p


def emissions(data, probabilities, smoothing):
    p = np.asarray(probabilities)
    if p.shape != (len(data['labels']), 8) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1.):
        raise ValueError('Require all aligned question-excluded word forecasts')
    support = np.ones((2, 8, core.N_OBS)); support[:, core.CURRENT == 1, -1] = 0.
    def counts(subject=None):
        result = .1 * support.copy()
        for i, path in enumerate(data['paths']):
            if not path or (subject is not None and data['subjects'][i] != subject):
                continue
            signal = core.observation(p[i], bool(data['answer_present'][i]))
            result[int(path == 'retry'), data['labels'][i], signal] += 1
        return result
    normalize = lambda x: x / x.sum(-1, keepdims=True)
    pooled = normalize(counts())
    return {subject: normalize(counts(subject) + smoothing * pooled) for subject in sorted(set(data['subjects']))}


def build(parent, forecaster, data, oof, smoothing, provenance):
    if (parent.fit_receipt['state_count'] != 8 or parent.fit_receipt['calibration_C'] != 1.
            or parent.fit_receipt['smoothing'] != smoothing
            or parent.fit_receipt['paired_questions'] != len(set(data['ids']))):
        raise ValueError('Require the matching unchanged C1 eight-state parent')
    if forecaster.variant != 'word12_C1' or forecaster.text.lowercase or smoothing not in (8., 32.):
        raise ValueError('Use only the approved case-sensitive word C1 and fixed smoothing')
    updated = emissions(data, oof, smoothing)
    dynamics = copy.deepcopy(parent.dynamics)
    for subject, d in dynamics.items():
        if type(d) is not core.Dynamics:
            raise ValueError('Preserve the exact original finite-horizon implementation')
        d.emission = updated[subject]
        for key in ('prior', 'transition', 'cost'):
            np.testing.assert_array_equal(getattr(d, key), getattr(parent.dynamics[subject], key))
        support = core.CURRENT[None, None, :] == core.AFTER[:, :, None]
        if np.any(d.transition[~support] != 0) or np.any(d.emission[:, core.CURRENT == 1, -1] != 0):
            raise ValueError('Changed hard current/continuation or missing-answer support')
    receipt = {**parent.fit_receipt, 'policy_module': __name__, 'observation_model': 'word12_C1',
        'temperature': 1., 'case_preserved': True, 'full_training_questions': len(set(data['ids'])),
        'changed': ['paid text/numeric joint forecast', 'question-excluded child emission table'],
        'preserved': ['prior', 'transition', 'cost', 'state support', 'finite Bellman backup', 'two-call horizon'],
        'provenance': provenance, 'sparse_cpu_cost': 'Measured separately; unchanged parameter-token proxy excludes CPU planning/linear algebra'}
    return Planner(parent.classifier, forecaster, dynamics, receipt)


def evaluate(row, tree, judges, model, settings, family):
    if not isinstance(model, Planner):
        raise ValueError('Use the registered sparse-word POMDP')
    model.head_receipts = []
    item = continuous.evaluate(row, tree, judges, model, settings, family)
    item.update(sparse_head_observations=list(model.head_receipts), sparse_head_calls=len(model.head_receipts),
                sparse_head_cpu_seconds=sum(v['cpu_seconds'] for v in model.head_receipts),
                sparse_extra_neural_calls=0)
    return item
