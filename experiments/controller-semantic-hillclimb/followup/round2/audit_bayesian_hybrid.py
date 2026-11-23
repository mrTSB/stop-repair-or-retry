"""Independently audit completed ensemble decisions, bills and selection.

Uses the already reused development cohort. Frozen neural weights are never
loaded and no forward/cloud/test access occurs. Dynamic programming checks
the subject-composition selector independently of its Pareto implementation.
"""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import audit_subject_checkpoint as independent
import development_bayesian_hybrid_v3 as runner

io,original = runner.io,runner.original


def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-8):raise ValueError('Independent numerical audit differs')


def check_items(items,rows,trees,judges):
    by_id = {r['id']:r for r in rows}
    if len(items) != 256 or {v['id'] for v in items} != set(by_id):raise ValueError('All256 questions required')
    for item in items:
        row,tree = by_id[item['id']],trees[item['id']]
        path = item['path']
        correct = int(original.policy.core.action_core.correct(tree[path],row['answer']))
        if item['correct'] != correct:raise ValueError('Saved correctness differs from frozen scorer')
        ancestry = ['']
        if path:
            parts = path.split('/')
            ancestry += ['/'.join(parts[:i]) for i in range(1,len(parts)+1)]
        if len(ancestry) > 3 or item['solver_calls'] != len(ancestry):raise ValueError('Changed solver horizon')
        solver_tokens = sum(tree[p]['input_tokens']+tree[p]['output_tokens'] for p in ancestry)
        if solver_tokens != item['solver_tokens']:raise ValueError('Missing or added solver token bill')
        queried = [d['path'] for d in item['decisions'] if d['queried']]
        if queried not in ([],['']):raise ValueError('Unexpected reviewer path')
        judge_tokens = sum(judges[row['id'],p]['input_tokens']+judges[row['id'],p]['output_tokens'] for p in queried)
        if item['judge_calls'] != len(queried) or judge_tokens != item['judge_tokens']:
            raise ValueError('Missing or added logical/physical-review token bill')
        seen,work = set(),0.
        for component in item['verifier_components']:
            key = tuple(component[k] for k in ('recipe_sha256','kind','path','input_sha256'))
            if key in seen or component['path'] not in ancestry:raise ValueError('Duplicate or unpaid observer charge')
            if component['kind'] == 'review' and component['path'] not in queried:raise ValueError('Unpaid review observer')
            seen.add(key)
            close(component['compute'],component['parameters_billions']*component['slots']/3.09)
            work += component['parameters_billions']*component['slots']
        close(item['compute'],(4*solver_tokens+3*judge_tokens+work)/3.09)
        close(item['total_parameter_compute'],(4*solver_tokens+35*judge_tokens+work)/3.09)
    return independent.stats(items)


def audit(folder):
    spec,done,summary = (io.read(folder/name) for name in ('registration.json','complete.json','summary.json'))
    if (done['registration_sha256'] != io.sha(folder/'registration.json') or done['test_used']
            or summary['fresh_validation'] or summary['schema'] != runner.SCHEMA):raise ValueError('Require completed reused-DEV search')
    io.verify({str(folder/k):v for k,v in done['digests'].items()})
    io.verify(spec['source_files'])
    args = SimpleNamespace(previous=Path(spec['previous']),output=folder)
    with runner.configured():
        bundle,prior,_,record = original.inputs(args)
    if record != spec:raise ValueError('Changed registered parent/cache/input evidence')
    rows,trees,judges,*_ = bundle
    candidates = io.read(folder/'candidate-subject-metrics.json')
    outputs = io.read(folder/'private-new-candidate-items.json')
    if len(candidates) != 192 or set(outputs) != {'E'+str(i) for i in range(156,192)}:
        raise ValueError('Preserve all36 new and156 old candidates')
    for candidate in candidates[156:]:
        items = list(outputs[candidate['key']].values())
        independent.equal(check_items(items,rows,trees,judges),candidate['candidate']['dev'])
        for subject in independent.SUBJECTS:
            panel = [v for v in items if v['subject'] == subject]
            expected = candidate['subjects'][subject]
            if len(panel) != expected['n'] or sum(v['correct'] for v in panel) != expected['correct']:
                raise ValueError('Subject metric differs from actual candidate outcomes')
            close(math.fsum(v['compute'] for v in panel)/len(panel),expected['mean_compute'])
    chosen = io.read(folder/'private-selected-items.json')
    measured = check_items(chosen,rows,trees,judges)
    independent.equal(measured,summary['C_candidate']['dev'])
    best,cost,_ = independent.optimum(candidates,prior['B']['dev']['mean_compute'])
    if best != measured['correct']:raise ValueError('Independent accuracy optimum differs')
    close(cost,measured['mean_compute'])
    target = summary['C_candidate']
    if io.sha(target['model_path']) != target['model_sha256']:raise ValueError('Changed serialized hybrid')
    if summary['rows'][:-1] != prior['rows'][:-1] or summary['B'] != prior['B']:
        raise ValueError('Never change the five baselines, A or stronger B')
    if original.prior.c80_targets.assess(summary['rows']) != summary['targets']:raise ValueError('Changed exact integer targets')
    return {'schema':'independently-audited-Bayesian-POMDP-hybrid-DEV-v1',
        'rows':summary['rows'],'targets':summary['targets'],'C':target,
        'prior_C':prior['C_candidate']['dev'],'offline_feature_extraction':summary['offline_feature_extraction'],
        'new_candidates_audited':36,'old_candidates_retained':156,'same_frozen_solver_and_reviewer':True,
        'all_per_question_correctness_and_component_bills_checked':True,'independent_dynamic_programming_selection_checked':True,
        'all_selected_runtime_decisions_reproduced':summary['all_selected_runtime_items_reproduced'],
        'source_sha256':io.sha(__file__),'complete_sha256':io.sha(folder/'complete.json'),
        'registration_sha256':io.sha(folder/'registration.json'),
        'reused_development_questions':256,'fresh_validation':False,'test_used':False,
        'limitation':'Exploratory selection on repeatedly reused DEV; proxy is parameter-weighted tokens, not measured FLOPs or wall time'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    with original.prior.guard.no_test_reads(),original.prior.guarded.no_model_calls():result = audit(args.run)
    if args.output.exists() and io.read(args.output) != result:raise ValueError('Preserve prior audit output')
    io.write(args.output,result)
    print({'C':result['C']['dev'],'targets':result['targets'],'independently_audited':True},flush=True)


if __name__ == '__main__':main()
