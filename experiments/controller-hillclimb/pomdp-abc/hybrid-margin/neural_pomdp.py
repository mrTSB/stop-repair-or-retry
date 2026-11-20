"""Use the learned verifier only as a noisy POMDP observation, charging every pass."""

import json
from pathlib import Path

import numpy as np

import difficulty_pomdp as base
from neural_verifier import Observer


def fit(rows,trees,judges,observer_directory,predictions,leaf=0,smoothing=32.,
        late_feedback=False,stage_review=True,contextual=False,transition_leaf=24,context_weight=.7):
    observer=Observer(str(Path(observer_directory).resolve()))
    recipe=json.loads((Path(observer_directory)/'recipe.json').read_bytes())
    slots_price=recipe['parameters']/1e9*recipe['max_length']/3.09
    entries=[(r,p) for r in rows for p in trees[r['id']]]
    scores=np.array([predictions[r['id'],'solver',p] for r,p in entries])
    extra={(r['id'],p):slots_price if p and '/' not in p and trees[r['id']][p]['answer'] else 0. for r,p in entries}
    model=base.fit(rows,trees,judges,leaf,smoothing,late_feedback,stage_review,
                   classifier_override=observer,scores_override=scores,extra_costs=extra)
    model.fit_receipt.update({'classifier_backend':'supervised MiniLM plus numeric features',
                             'grouped_folds':recipe['folds'],
                             'observer_weights_sha256':recipe['weights_sha256'],
                             'observer_parameters':recipe['parameters'],'observer_max_length':recipe['max_length']})
    if contextual:
        from contextual_pomdp import fit as context_fit
        model=context_fit(rows,trees,judges,leaf,smoothing,late_feedback,stage_review,
                          transition_leaf,context_weight,initial_model=model)
    return model


def evaluate(row,tree,judges,model,settings,method):
    if method not in ('A','C','always','never'):raise ValueError('The verifier is not an LLM controller B')
    evaluator=base.evaluate
    if settings.get('contextual_transitions',False):
        from contextual_pomdp import evaluate as contextual
        evaluator=contextual
    result=evaluator(row,tree,judges,model,settings,method)
    last_observation=result['decisions'][-1]['path']
    tokens,calls,parameters=model.classifier.charges(row,tree,last_observation)
    # The implementation pads every pass to this length. Charge those slots,
    # rather than only non-padding text tokens, for a conservative compute proxy.
    slots=calls*model.classifier._recipe['max_length']
    extra=slots*parameters/3.09
    result.update({'verifier_tokens':tokens,'verifier_calls':calls,'verifier_token_slots':slots,
                   'verifier_parameters_billions':parameters})
    result['compute']+=extra;result['total_parameter_compute']+=extra
    return result
