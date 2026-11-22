"""Complete16 POMDP with a paid correctness sensor for valid science answers."""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

import regularized_completion_pomdp as previous
import completion_subject_pomdp as routed
import semantic_current_belief as correction
from science_review_critic import margin

core = previous.core


@dataclass
class Planner(previous.Planner):
    semantic: object = None
    calibrator: object = None

    def predicted(self, row, path, tree):
        p = super().predicted(row, path, tree)
        if not tree[path]['answer']: return p
        safe = {**row, 'answer': ''}; paid = {k: tree[k] for k in ('', path)}
        raw = self.semantic.values(safe, path, paid)[0]
        x = correction.features([{'subject': row['subject'], 'path': path}], p[None], [float(margin(raw)/self.semantic.temperature)])
        current = self.calibrator.predict_proba(x)[:, 1]
        return correction.update(p[None], current, [True])[0]


def refit(base, rows, trees, records, probability, semantic, calibrator, provenance):
    """Retain768-question transitions; refit sensor emissions on256 OOF groups."""
    expected = [(r['id'], p) for r in rows for p in ('', 'repair', 'retry')]
    if [(r['id'], r['path']) for r in records] != expected: raise ValueError('Sensor forecast order changed')
    lookup = dict(zip(expected, probability)); support = np.ones((2, 16, core.N_OBS))
    support[:, core.CURRENT == 1, 24:] = 0.
    semantic_price = semantic._recipe['parameters']/1e9*512/3.09
    def counts(group):
        emission = .1*support.copy(); prices = np.zeros((2, 16)); number = np.zeros_like(prices)
        for row in group:
            qid = row['id']; tree = trees[qid]
            root = core.state(*[core.plain.correct(tree[p], row['answer']) for p in ('', 'repair', 'retry', 'complete')])
            for action, path in enumerate(('repair', 'retry')):
                state = core.state(*[core.plain.correct(tree[p], row['answer']) for p in (path, path+'/repair', path+'/retry', path+'/complete')])
                observed = core.observation(lookup[qid, path], bool(tree[path]['answer']))
                emission[action, state, observed] += 1
                prices[action, root] += semantic_price*bool(tree[path]['answer']); number[action, root] += 1
        return emission, prices, number
    normalize = lambda x: x/x.sum(-1, keepdims=True)
    pooled = counts(rows); dynamics = copy.deepcopy(base.dynamics)
    pooled_price = np.divide(pooled[1], pooled[2], out=np.full_like(pooled[1], semantic_price/2), where=pooled[2] > 0)
    for subject, d in dynamics.items():
        local = counts([r for r in rows if r['subject'] == subject])
        d.emission = normalize(local[0]+base.fit_receipt['smoothing']*normalize(pooled[0]))
        d.cost[0, :2] += (local[1]+4*pooled_price)/(local[2]+4)
        np.testing.assert_array_equal(d.prior, base.dynamics[subject].prior)
        np.testing.assert_array_equal(d.transition, base.dynamics[subject].transition)
        np.testing.assert_array_equal(d.cost[1], base.dynamics[subject].cost[1])
    receipt = {**base.fit_receipt, 'semantic_current_correction': True, 'sensor_calibration_questions': len(rows),
        'sensor_emission_questions': len(rows), 'transition_questions': 768, 'sensor_folds_nested': True,
        'extra_observer': 'One paid actual-trimmed V3 pass for each valid observed science answer',
        'planning_sensor_cost': 'Conservative512-slot additional child price, conditional on TRAIN validity; terminal actions unchanged',
        'original_transition_and_prior_retained': True, 'provenance': provenance}
    return Planner(copy.deepcopy(base.classifier), copy.deepcopy(base.forecaster), dynamics, receipt,
                   base.feature_family, semantic, calibrator)


def evaluate_science(row, tree, judges, model, settings, family):
    if family != 'A': raise ValueError('No LLM observations in controller A')
    item = core.evaluate(row, tree, {}, model, settings, family); safe = {**row, 'answer': ''}; ledger = []
    for decision in item['decisions']:
        path = decision['path']; paid = {k: tree[k] for k in ('', path)}
        for name, observer in [('primary', model.classifier), *([('semantic', model.semantic)] if tree[path]['answer'] else [])]:
            values = observer.values(safe, path, paid)
            entry = observer.entry(safe, path, paid); slots = int(values[3]); params = observer._recipe['parameters']/1e9
            ledger.append({'component': name, 'path': path, 'input_sha256': hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest(),
                'recipe_sha256': hashlib.sha256((Path(observer.directory)/'recipe.json').read_bytes()).hexdigest(),
                'token_slots': slots, 'parameters_billions': params, 'compute': slots*params/3.09})
    primary = sum(v['compute'] for v in ledger if v['component'] == 'primary')
    if not np.isclose(primary, item['verifier_token_slots']*item['verifier_parameters_billions']/3.09): raise ValueError('Primary cost changed')
    slots = sum(v['token_slots'] for v in ledger); work = sum(v['compute'] for v in ledger)
    item.update(verifier_components=ledger, verifier_tokens=slots, verifier_token_slots=slots,
                verifier_calls=len(ledger), verifier_parameters_billions=work*3.09/slots)
    item['compute'] += work-primary; item['total_parameter_compute'] += work-primary
    return item


SubjectPlanner = routed.Planner


def evaluate(row, tree, judges, model, settings, family):
    if family != 'A': raise ValueError('Science correctness sensor is not an LLM reviewer')
    if row['subject'] == 'math': return routed.evaluate(row, tree, {}, model, settings, family)
    return evaluate_science(row, tree, {}, model.science, model.science_settings, family)
