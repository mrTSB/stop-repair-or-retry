"""Twenty-four prespecified LLM-only rules with the new terminal action.

The existing strongest B remains a separate retained candidate. Each new
science policy pays the same strong root review even when it stops or chooses
Complete. Only its action/confidence/agreement and current validity guide the
policy. Math is supplied as the unchanged retained B's archived outcomes.
No collector, learned model, neural features or future probability features.
"""
from collections import Counter
from itertools import product

import matched_review_b as original

SETTINGS=[{'root_stop':stop,'first_repair':style,'ordinary_completion':complete}
          for stop,style,complete in product(original.ROOT_STOPS,('ordinary','guided'),('always','invalid'))]


def policy(row,tree,judges,settings):
    if row['subject'] not in ('chemistry','physics') or settings not in SETTINGS:
        raise ValueError('Use the prespecified24 science LLM rules only')
    qid=row['id']
    if (qid,'') not in judges: raise original.MissingObservation('judge','')
    judge=judges[qid,''];original.validate_review_charge(judge)
    if any(call[field]<0 for call in judge['calls'] for field in ('input_tokens','output_tokens')):
        raise ValueError('A physical review phase cannot have negative cost')
    rule,action=settings['root_stop'],judge['action']
    if rule=='agreement': action='stop' if original.equal(judge['answer'],tree['']['answer']) else 'repair'
    elif rule!='action': action='stop' if judge['p_correct']>=rule else (action if action!='stop' else 'repair')
    if action not in ('stop','repair','retry'): raise ValueError('Unknown LLM root action')
    path,visited='', ['']
    # Invalid submitted text may be completed once even when the LLM says Stop.
    if action=='stop' and not tree['']['answer']: action='complete'
    decisions=[{'path':'','action':action,'queried':True}]
    if action!='stop':
        path=(settings['first_repair'] if action=='repair' else action)
        if path=='ordinary': path='repair'
        if path not in tree: raise original.MissingObservation('solver',path)
        visited.append(path)
        if path!='complete':
            if path=='guided':
                next_action='stop' if original.equal(judge['answer'],tree[path]['answer']) else 'reuse'
            else:
                next_action='complete' if settings['ordinary_completion']=='always' or not tree[path]['answer'] else 'stop'
            decisions.append({'path':path,'action':'repair' if next_action=='reuse' else next_action,'queried':False})
            if next_action!='stop':
                path+='/'+next_action
                if path not in tree: raise original.MissingObservation('solver',path)
                visited.append(path)
    item=original.outcome(row,tree,judges,path,visited,[''],decisions)
    item.update(judge_physical_calls=judge['physical_api_calls'],verifier_calls=0,
        verifier_tokens=0,verifier_token_slots=0,verifier_parameters_billions=0.,
        completion_calls=int(path.endswith('complete')),completion_is_new_action=True)
    if item['solver_calls']>3 or item['judge_calls']!=1 or path=='guided/complete':
        raise ValueError('Expanded B violated the exact action availability/horizon')
    return item


def evaluate_panel(rows,trees,judges,math_outcomes,settings):
    if (len(rows)!=256 or len({r['id'] for r in rows})!=256
            or Counter(r['subject'] for r in rows)!={'math':128,'chemistry':64,'physics':64}
            or set(math_outcomes)!={r['id'] for r in rows if r['subject']=='math'}):
        raise ValueError('All256 DEV questions and unchanged128math B outcomes are required')
    if any(math_outcomes[qid]['id']!=qid or math_outcomes[qid]['subject']!='math' for qid in math_outcomes):
        raise ValueError('Retained math outcome identity changed')
    return [math_outcomes[r['id']] if r['subject']=='math' else policy(r,trees[r['id']],judges,settings) for r in rows]


def compare(candidates,retained):
    if len(candidates)!=24 or not retained or {tuple(sorted(c['settings'].items())) for c in candidates}!={tuple(sorted(s.items())) for s in SETTINGS}:
        raise ValueError('Keep all24 new candidates and all supplied original B winners')
    for c in [*candidates,*retained]:
        if c['dev']['n']!=256: raise ValueError('Compare only the same full256 panel')
    winner=max([*retained,*candidates],key=lambda c:(c['dev']['accuracy'],-c['dev']['mean_compute']))
    return {'B':winner,'new_best_B':max(candidates,key=lambda c:(c['dev']['accuracy'],-c['dev']['mean_compute'])),
        'retained_original_B':retained,'search':candidates,'new_action_protocol':True,
        'method':'LLM-only rules; no POMDP or neural observer','original_five_baselines_unchanged':True,
        'fresh_validation':False,'test_used':False}
