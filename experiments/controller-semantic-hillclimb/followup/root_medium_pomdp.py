"""POMDP hybrid allocating at most one deeper LLM review at the root.

After a paid root review the same compact solver may repair with an answer-first
packet, or retry independently. The second decision uses the POMDP alone.
"""

import numpy as np

import dynamic_grounded_pomdp as perception
import difficulty_pomdp as core
from context_pomdp import REVIEW_WIDTH, mentor_comparison, outcome
from medium_pomdp import merged_observations, root_mentor
from scoring import correct


def training_observations(metadata, outputs, additional, predictions):
    keep = [i for i, r in enumerate(metadata)
            if not r['path'].startswith('guided') and not r['path'].endswith('/reviewed')
            and not (r['kind'] == 'original-review' and r['path'] == '')]
    return merged_observations([metadata[i] for i in keep],
                               {k: outputs[k][keep] for k in ('state', 'plain', 'guided')},
                               additional, predictions)


def fit(rows, trees, judges, directory, metadata, outputs, smoothing=32.,
        semantic_review=True, late=False, context_weight=0.):
    model = perception.fit(rows, trees, judges, directory, metadata, outputs,
                            smoothing=smoothing, late=False, semantic_review=semantic_review,
                            context_weight=0.)
    # A feedback repair is a different transition kernel from a plain repair.
    # For sparse review signals, shrink toward the observed reviewed kernel
    # rather than assuming feedback behaves like an unreviewed solver call.
    lookup = {(r['id'], r['kind'], r['path']): v for r, v in zip(metadata, outputs['state'])}
    for subject, d in model.dynamics.items():
        counts = np.zeros_like(d.reviewed_transition[0])
        pooled = 4. * d.transition[0]
        for row in rows:
            qid = row['id']
            if row['subject'] != subject or (qid, '') not in judges:
                continue
            tree = trees[qid]
            hard = int(np.mean([correct(tree[p], row['answer']) for p in core.FIXED]) < .6)
            state = 2 * hard + int(correct(tree[''], row['answer']))
            judge = judges[qid, '']
            signal = (perception.review_signal(perception.softmax(lookup[qid, 'original-review', ''],
                                                                  model.classifier.temperature), judge, tree[''])
                      if semantic_review else core.encode(judge, tree['']))
            for action, path in enumerate(('guided', 'retry')):
                if path not in tree:
                    continue
                successor = 2 * hard + int(correct(tree[path], row['answer']))
                counts[signal, action, state, successor] += 1
                pooled[action, state, successor] += 1
        pooled /= pooled.sum(-1, keepdims=True)
        counts += smoothing * pooled[None]
        d.reviewed_transition[0] = counts / counts.sum(-1, keepdims=True)
    root_mentor(model, rows, trees, judges, metadata, outputs, smoothing)
    model.fit_receipt.update(
        root_medium_review=True, maximum_logical_reviews=1,
        review_availability='Root only; no paid LLM observation at the second decision',
        root_review='Independent1024bounded thought and512forced JSON; all phases charged',
        guided_format='Final first, at most35justification words,128output tokens',
        plain_actions='Original repair/retry prompts and256token continuation cap',
        reviewed_kernel_prior='Hierarchical reviewed transitions, pooled pseudocount4 from ordinary transitions',
        mentor_estimation='Matched192root medium reviews and their actual compact continuations')
    return model


def evaluate(row, tree, judges, model, settings, family):
    if family != 'C' or settings.get('plan_reviews', False):
        raise ValueError('This hybrid has one optional root review and no future paid reviews')
    d = model.dynamics[row['subject']]
    penalty = settings['penalty'] * (settings.get('science_weight', 1.) if row['domain'] == 'science' else 1.)
    continuous = settings.get('continuous_filter', False)
    threshold = settings.get('threshold', 0.) + 1e-9
    reuse = settings.get('reuse_review', True)
    observed = model.observed(row, '', tree)
    belief = d.update(d.prior, model.likelihood(row, '', tree) if continuous else d.initial[:, observed])
    path, visited, acquired, decisions = '', [''], [], []
    for remaining in (2, 1):
        stage = 2 - remaining
        condition = model.observed(row, path, tree)
        transition = d.transition[stage]
        values = d.values(belief, remaining, penalty, transition)
        query, information, judge = False, None, None
        if stage == 0:
            expectation = d.information_value(belief, remaining, penalty, condition, reuse)
            information = float(expectation - values.max())
            query = information > threshold
        if query:
            if (row['id'], '') not in judges:
                raise core.MissingObservation('judge', '')
            judge = judges[row['id'], '']
            signal = model.review_signal(row, '', tree, judge)
            acquired.append('')
            belief = d.update(belief, d.likelihood(0, condition)[:, signal])
            transition = d.reviewed_transition[0, signal]
            values = d.values(belief, remaining, penalty, transition, True, reuse)
        action = ('stop', 'repair', 'retry')[int(values.argmax())]
        decisions.append({'path': path, 'belief': belief.tolist(), 'action': action,
                          'queried': bool(query), 'information_value': information,
                          'root_medium_review': True, 'continuous_filter': continuous})
        if action == 'stop':
            break
        edge = 'guided' if query and action == 'repair' else action
        path = f'{path}/{edge}' if path else edge
        if path not in tree:
            raise core.MissingObservation('solver', path)
        visited.append(path)
        if remaining == 1:
            break
        observed = model.observed(row, path, tree)
        comparison = mentor_comparison(judge, tree[path]) if query and reuse else None
        if continuous:
            likelihood = model.likelihood(row, path, tree)
            if comparison is not None:
                likelihood = likelihood * d.mentor[observed, :, comparison]
            belief = d.update(belief @ transition[('repair', 'retry').index(action)], likelihood)
        else:
            belief = d.next_belief(belief, transition, ('repair', 'retry').index(action),
                                   observed, 0, query, comparison)
    result = outcome(row, tree, judges, path, visited, acquired, decisions)
    tokens, calls, slots, parameters = model.classifier.charges(
        row, tree, judges, decisions, include_reviews=model.review_observer is not None)
    extra = parameters * slots / 3.09
    result.update(verifier_tokens=tokens, verifier_calls=calls, verifier_token_slots=slots,
                  verifier_parameters_billions=parameters)
    result['compute'] += extra
    result['total_parameter_compute'] += extra
    return result
