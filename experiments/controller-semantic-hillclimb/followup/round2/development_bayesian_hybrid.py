"""Registered cache-only C ensemble search, retaining all earlier C candidates.

Thirty-six fixed ensemble rules use two TRAIN-fitted POMDPs. Reused DEV selects
observed-subject experts under the unchanged strongest B compute ceiling.
No model fitting, neural forward, cloud call or fresh-test read is permitted.
"""
import argparse
from contextlib import ExitStack
from itertools import product
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

from threadpoolctl import threadpool_limits

import development_subject_experts as prior
import bayesian_model_hybrid as policy

io,selection,canonical = prior.io,prior.selection,prior.canonical
SCHEMA = 'Bayesian-model-uncertainty-POMDP-hybrid-reused-DEV-v1'
PARENTS = ('smooth8-review1','smooth32-review1')
WEIGHTS = (.25,.5,.75)
PENALTIES = (0.,.01,.05,.1)
THRESHOLDS = (0.,.02,.08)


def inputs(args):
    inherited = io.read(args.previous/'registration.json')
    saved = SimpleNamespace(**{k:Path(v) for k,v in inherited['paths'].items()})
    bundle = prior.inputs(saved)
    if bundle[-1] != inherited: raise ValueError('Changed prior subject-expert registration')
    completed = io.read(args.previous/'complete.json')
    io.verify({str(args.previous/k):v for k,v in completed['artifacts'].items()})
    summary = io.read(args.previous/'summary.json')
    if summary['B']['dev']['correct'] != 179 or summary['C_candidate']['dev']['correct'] != 177:
        raise ValueError('Retain the independently audited179/177 starting comparison')
    old = io.read(saved.canonical/'registration.json')
    source = Path(old['paths']['source'])
    models = {name:source/'models'/(name+'.pkl') for name in PARENTS}
    record = {'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),
        'prior_registration_sha256':io.sha(args.previous/'registration.json'),
        'prior_complete_sha256':io.sha(args.previous/'complete.json'),
        'parent_models':{key:{'path':str(path.resolve()),'sha256':io.sha(path)} for key,path in models.items()},
        'weights':list(WEIGHTS),'penalties':list(PENALTIES),'thresholds':list(THRESHOLDS),
        'new_candidates':36,'old_candidates_retained':156,'hidden_state':'Static model index times original8-state mentor state; original unqueried A retained',
        'planning':'Sum model-state alpha values before maximization for every shared future observation and action',
        'billing':'All distinct recipe/input observer forwards, all solver tokens and every physical review phase; exact duplicates charged once',
        'selection':'Exact per-subject Pareto product over192 C candidates, highest accuracy then lower compute strictly below unchanged strongest B',
        'fresh_validation':False,'test_used':False,'reused_DEV':True,'new_neural_calls':0,'new_cloud_calls':0,'training':False,
        'all_five_baselines_and_strongest_A_B_unchanged':True,'previous':str(args.previous.resolve()),'output':str(args.output.resolve())}
    return bundle,summary,source,record


def replay(args,bundle,summary,source,record):
    rows,trees,judges,_,caches,old_cs,old_paired,*_ = bundle
    cs = list(old_cs)
    paired = {v['key']:{r['id']:r for r in old_paired[v['candidate']['original_candidate_index']].values()} for v in old_cs}
    folder = args.output/'models';folder.mkdir()
    for weight in WEIGHTS:
        key = 'ensemble-'+str(weight)
        parents = [pickle.loads((source/'models'/(name+'.pkl')).read_bytes()) for name in PARENTS]
        model = policy.Planner(parents,[weight,1-weight],{'schema':SCHEMA,'parent_models':record['parent_models'],
            'prior_weights':[weight,1-weight],'TRAIN_fitted_parents':True,'new_training':False})
        # Serialize unloaded parents before any DEV cache is attached.
        target = folder/(key+'.pkl');io.atomic_bytes(target,pickle.dumps(model,protocol=5))
        with ExitStack() as stack:
            for child in parents: stack.enter_context(canonical.cached_model(child,caches))
            for penalty,threshold in product(PENALTIES,THRESHOLDS):
                index = len(cs)
                setting = {'penalty':penalty,'threshold':threshold}
                items = [policy.evaluate(r,trees[r['id']],judges,model,setting,'C') for r in rows]
                if any(v['solver_calls'] > 3 or v['judge_calls'] > 1 for v in items): raise ValueError('Changed horizon')
                candidate = {'model':key,'policy_module':'bayesian_model_hybrid','settings':setting,
                    'model_path':str(target.resolve()),'model_sha256':io.sha(target),'original_candidate_index':index,
                    'dev':selection.metrics(items)}
                tag = 'E'+str(index)
                for row,item in zip(rows,items):item['subject'] = row['subject']
                subjects = {s:selection.metrics([v for v in items if v['subject'] == s]) for s in selection.SUBJECTS}
                cs.append({'key':tag,'candidate':candidate,'subjects':subjects})
                paired[tag] = {v['id']:v for v in items}
                print({'candidate':index,'weight':weight,'settings':setting,'dev':candidate['dev']},flush=True)
    selected,fronts = selection.best_C(cs,summary['B']['dev']['mean_compute'])
    if selected is None: raise ValueError('No qualifying C; preserve old comparison')
    choices,value = selected
    if value['correct'] < summary['C_candidate']['dev']['correct']: raise ValueError('Never replace stronger prior C')
    experts = {s:choices[s]['candidate'] for s in selection.SUBJECTS}
    raw = {}
    for expert in experts.values():
        key = expert['model']
        path = Path(expert.get('model_path',source/'models'/(key+'.pkl')))
        if io.sha(path) != expert['model_sha256']: raise ValueError('Changed selected model')
        raw[key] = pickle.loads(path.read_bytes())
    runtime = policy.RoutedPlanner(raw,experts,{'schema':SCHEMA,'route_visibility':'Observed subject only',
        'new_neural_calls':0,'new_cloud_calls':0,'fresh_validation':False,'test_used':False})
    target = args.output/'hybrid.pkl';io.atomic_bytes(target,pickle.dumps(runtime,protocol=5))
    selected_items = []
    with ExitStack() as stack:
        for child in raw.values():
            for parent in child.models if isinstance(child,policy.Planner) else [child]:
                stack.enter_context(canonical.cached_model(parent,caches))
        for row in rows:
            item = policy.routed(row,trees[row['id']],judges,runtime,{},'C')
            expected = paired[choices[row['subject']]['key']][row['id']]
            if any(item[k] != expected[k] for k in ('path','correct','compute')):
                raise ValueError('Final runtime must reproduce selected decisions and full bills exactly')
            selected_items.append(item)
    prior.same(selection.metrics(selected_items),value)
    table = [r for r in summary['rows'] if r['method'] != 'C']+[{'method':'C',**value}]
    result = {'schema':SCHEMA,'rows':table,'C_candidate':{'policy_module':'bayesian_model_hybrid.routed',
        'model_path':str(target.resolve()),'model_sha256':io.sha(target),'experts':experts,'dev':value},
        'prior_C':summary['C_candidate'],'B':summary['B'],'targets':prior.c80_targets.assess(table),
        'frontier_sizes':fronts,'new_candidates':36,'old_candidates_retained':156,
        'all_selected_runtime_items_reproduced':True,'fresh_validation':False,'test_used':False,
        'new_neural_calls':0,'new_cloud_calls':0,'reused_DEV':True,'candidate_until_independent_audit':True}
    io.write(args.output/'private-selected-items.json',selected_items)
    io.write(args.output/'candidate-subject-metrics.json',cs)
    io.write(args.output/'private-new-candidate-items.json',{k:v for k,v in paired.items() if k.startswith('E')})
    io.write(args.output/'summary.json',result)
    io.verify(record['source_files'])
    if inputs(args)[-1] != record: raise ValueError('Changed registered inputs during replay')
    io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
        'digests':{str(p.relative_to(args.output)):io.sha(p) for p in args.output.rglob('*') if p.is_file() and not p.name.startswith('.')},
        'fresh_validation':False,'test_used':False})
    print({'C_candidate':value,'experts':experts,'targets':result['targets']},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','replay'))
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    with prior.guard.no_test_reads(),prior.guarded.no_model_calls(),threadpool_limits(limits=1):
        bundle,summary,source,record = inputs(args)
        if args.phase == 'register':
            args.output.mkdir(parents=True,exist_ok=False)
            io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json'),'new_candidates':36},flush=True)
        else:
            if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                    or io.read(args.output/'registration.json') != record): raise ValueError('Changed ensemble search registration')
            with io.lock(args.output/'.replay.lock'):
                if {p.name for p in args.output.iterdir()} != {'registration.json','.replay.lock'}: raise ValueError('Preserve previous replay')
                replay(args,bundle,summary,source,record)


if __name__ == '__main__': main()
